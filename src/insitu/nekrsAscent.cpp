#include "nrs.hpp"
#include "nekrsAscent.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

#include <sys/sysinfo.h>
#include <queue>

std::unique_ptr<NekrsAscent> NekrsAscent::singleton_;

NekrsAscent::NekrsAscent() {
}

NekrsAscent::~NekrsAscent() {
  if(mMode == "asynchronous") {
    // Signal the worker thread to stop
    {
      std::lock_guard<std::mutex> lock(mQueueMutex);
      mShouldStopThread = true;
    }
    mQueueCondition.notify_one();
    if (mWorkerThread.joinable()) {
      mWorkerThread.join(); // Ensure the thread finishes cleanly
    }
  }
  mAscent.close(); // Close Ascent instance to clean up
}

size_t NekrsAscent::calculateTotalMemoryNeeded(dlong Nlocal) const {
  // Calculate memory needed for velocity, pressure, and scalars
  size_t totalMem = 3 * Nlocal * sizeof(dfloat);  // vel_x, vel_y, vel_z
                                                  // pressure
  if(mNrs->o_P.ptr()) {
    totalMem += Nlocal * sizeof(dfloat); // pressure
  }

  totalMem += mNrs->cds->NSfields * Nlocal * sizeof(dfloat); // scalars
  return totalMem;	
}

// Define the static method to get the singleton instance
NekrsAscent& NekrsAscent::getInstance() {
  static std::once_flag flag;
  std::call_once(flag, [](){
      singleton_.reset(new NekrsAscent());
      });
  return *singleton_;
}

// Function to check if there is enough available memory
bool NekrsAscent::hasEnoughMemory(size_t requiredMemory) const {
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return info.freeram >= requiredMemory;
  }
  return false;
}

void NekrsAscent::finalize() {
  if(mMode == "synchronous") { 
    printStats();
    return;
  }

  // Signal thread to stop
  {
    std::lock_guard<std::mutex> lock(mQueueMutex);
    mShouldStopThread = true;
  }
  mQueueCondition.notify_all();

  // Wait for thread to complete
  if (mWorkerThread.joinable()) {
    mWorkerThread.join();
  }

  printStats();
}


void NekrsAscent::initializeAscent() {
  static bool initialized = false;
  if (!initialized) {
    MPI_Comm ascent_comm;

    // Split communicator based on the task ID
    MPI_Comm_dup(MPI_COMM_WORLD, &ascent_comm);

    conduit::Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(ascent_comm);

    const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;
    if (verbose) {
      ascent_opts["ascent_info"] = "verbose";
      ascent_opts["messages"] = "verbose";
    }

    mAscent.open(ascent_opts);
    initialized = true;
  }
}

std::vector<int> NekrsAscent::parseStringToVector(const std::string& input) {
    std::vector<int> result;
    std::string cleanedInput = input;

    // Remove '[' and ']' characters from the string
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '['), cleanedInput.end());
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), ']'), cleanedInput.end());

    std::stringstream ss(cleanedInput);
    std::string token;

    // Split the cleaned string by ',' and parse each part into an integer
    while (std::getline(ss, token, ',')) {
        result.push_back(std::stoi(token));  // Convert the token to an integer
    }

    return result;
}

void NekrsAscent::printStats() const { //TODO: try to extract img info??
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //const double tInsituTransferTime = platform->timer.query("insituTransferData", "DEVICE:MAX");
  double tInsituTransferTime, tAscentPublishDataTime, tAscentExecuteDataTime;
  MPI_Reduce(&mElapsedDataTransferTime, &tInsituTransferTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mElapsedAscentPublishTime, &tAscentPublishDataTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mElapsedAscentExecuteTime, &tAscentExecuteDataTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "======= INSITU TIMINGS =======" << std::endl;
    std::cout << "Total Data Transfer time from NekRS to ASCENT: " << tInsituTransferTime << std::endl;
    std::cout << "ASCENT Publish time: " << tAscentPublishDataTime << std::endl;
    std::cout << "ASCENT Execute time (rendering): " << tAscentExecuteDataTime << std::endl;
    //platform->timer.printStatEntry("    insituTransferData       ", "insituTransferData", "DEVICE:MAX", tInsituTransferTime);
  }
}

void NekrsAscent::run(const double time, const int tstep) { 
  if(mInterval < 1 || tstep % mInterval !=0) return;

  // First calculate how much memory will be needed
  size_t totalMem = calculateTotalMemoryNeeded(mMesh->Nlocal);
  if(!hasEnoughMemory(totalMem)) {
    std::cout <<  "rank: " << platform->comm.mpiRank << "Not enough memory, skipping step " << tstep << std::endl;
    return;
  }

  dlong nCells = mMesh->Nelements * (mMesh->Nq - 1) * (mMesh->Nq - 1) * (mMesh->Nq - 1);
  dlong nVertices = nCells * 8;
  cds_t *cds = mNrs->cds;

  platform->timer.tic("insituTransferData", 1);
  double tStart = MPI_Wtime();
  auto sim_data = std::make_shared<SimulationData>(mMesh, mNrs, mMode, time, tstep);
  double tEnd = MPI_Wtime();
  mElapsedDataTransferTime += (tEnd - tStart);
  platform->timer.toc("insituTransferData");

  if(mMode == "synchronous") {
    ascentPublishExecute(sim_data->getConduitData());
  } else { 
    {
      std::lock_guard<std::mutex> lock(mQueueMutex);
      mDataQueue.push(sim_data);
    }
    mQueueCondition.notify_one(); // Notify the worker thread of new data
  }
}

void NekrsAscent::setAffinity(pthread_t threadID) {
  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  // Create a subcommunicator for all ranks on the same node
  MPI_Comm localcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localcomm);

  int localRank, localSize;
  MPI_Comm_rank(localcomm, &localRank);
  MPI_Comm_size(localcomm, &localSize);

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  int num_cpus_assigned = 2; // One for main thread, one for the thread checking the queue
  if(mNekrsAffinity.empty() || mAscentAffinity.empty()) {
    // Get the affinity mask for the calling process (rank)
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
      perror("sched_getaffinity");
    }

    // Count how many CPUs are set in cpuset
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
      if (CPU_ISSET(cpu, &cpuset)) {
        num_cpus_assigned++;
      }
    }
  }

  int cpus_per_rank = num_cpus_assigned;
  int main_core_id = localRank * cpus_per_rank;
  int thread_core_id = localRank * cpus_per_rank;
  
  if(!mNekrsAffinity.empty()) {
    std::vector<int> affinity = parseStringToVector(mNekrsAffinity);
    main_core_id = affinity[localRank];

  }

  if(!mAscentAffinity.empty()) {
    std::vector<int> affinity = parseStringToVector(mAscentAffinity);
    thread_core_id = affinity[localRank];
  }

  std::cout << "World Rank: "<< worldRank << " Local Rank: " << localRank << " NekAscent (concurrent): Number of CPUs assigned to this MPI rank: " << num_cpus_assigned << std::endl;
  std::cout << "World Rank: "<< worldRank << " Local Rank: " << localRank << " NekAscent (concurrent): Setting affinity to main thread: " << main_core_id << " and while queue thread: " << thread_core_id << std::endl;

  // Create a CPU set and add the chosen core to it.
  cpu_set_t main_mask, thread_mask;
  CPU_ZERO(&main_mask);
  CPU_ZERO(&thread_mask);
  CPU_SET(main_core_id, &main_mask);
  CPU_SET(thread_core_id, &thread_mask);

  // Attempt to set affinity for this process (the main thread).
  if (sched_setaffinity(0, sizeof(main_mask), &main_mask) != 0) {
    std::cerr << "Error setting main thread affinity" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (pthread_setaffinity_np(threadID, sizeof(cpu_set_t), &thread_mask) != 0) {
    std::cerr << "Error setting queue thread affinity" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
}

void NekrsAscent::setup(nrs_t *nrs, int interval, std::string mode, 
                        std::string nekrsAffinity, std::string ascentAffinity) {
  // setup aide to create fields list from active fields
  mNrs = nrs;
  mMode = mode;
  mInterval = interval;
  mNekrsAffinity = nekrsAffinity;
  mAscentAffinity = ascentAffinity;

  (mNrs->cds) ? mMesh = mNrs->cds->mesh[0] : mMesh = mNrs->meshV; // cht
  if(mode == "asynchronous") {
    // Initialize the worker thread that will handle rendering tasks
    mWorkerThread = std::thread(&NekrsAscent::workerThreadFunc, this);
    // Set thread affinity
    pthread_t threadID = mWorkerThread.native_handle();
    setAffinity(threadID);
  }
  initializeAscent();
}

void NekrsAscent::waitForCompletion() {
  if(mMode == "synchronous") return;
  std::unique_lock<std::mutex> lock(mQueueMutex);
  mQueueCondition.wait(lock, [this] { return mDataQueue.empty(); });
}

void NekrsAscent::workerThreadFunc()
{
  while (true) {
    std::shared_ptr<SimulationData> sim_data;

    // Scope for lock while checking queue
    {
      std::unique_lock<std::mutex> lock(mQueueMutex);
      mQueueCondition.wait(lock, [this] { 
          return !mDataQueue.empty() || mShouldStopThread; 
          });

      if (mShouldStopThread && mDataQueue.empty()) {
        break; // Exit if stopping and queue is empty
      }

      if (!mDataQueue.empty()) {
        sim_data = mDataQueue.front();
        mDataQueue.pop();
      }
    } // Lock released here

    // Process data outside of lock
    if (sim_data) {
      ascentPublishExecute(sim_data->getConduitData());
      // Notify waitForCompletion that we processed an item
      mQueueCondition.notify_all();
    }
  }
}

void NekrsAscent::ascentPublishExecute(const conduit::Node& mesh) {
  double tStart = MPI_Wtime();
  mAscent.publish(mesh);
  double tEnd = MPI_Wtime();
  mElapsedAscentPublishTime += (tEnd - tStart);

  conduit::Node actions;

  tStart = MPI_Wtime();
  mAscent.execute(actions);
  tEnd = MPI_Wtime();
  mElapsedAscentExecuteTime += (tEnd - tStart);   
}

