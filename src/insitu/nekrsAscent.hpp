#ifndef NEKRS_ASCENT_HPP
#define NEKRS_ASCENT_HPP

/*
   insitu visualizatoin via Ascent 
   */

#include <vector>
#include <tuple>
#include <queue>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "nrs.hpp"
#include <ascent.hpp>
#include <mpi.h>

#include "simulationData.hpp"

class NekrsAscent
{
  protected:
    NekrsAscent();  // Constructor

    // Private static instance pointer
    static std::unique_ptr<NekrsAscent> singleton_;

  public:
    using Field = std::tuple<std::string, occa::memory, dlong>;
    using Fields = std::vector<Field>;

    static NekrsAscent& getInstance();  // Method to get the singleton instance

    /**
     * Singletons should not be cloneable.
     */
    NekrsAscent(const NekrsAscent &other) = delete;
    /**
     * Singletons should not be assignable.
     */
    NekrsAscent& operator=(const NekrsAscent &) = delete;


    ~NekrsAscent(); // Destructor

    // Ascent Publish and Execute
    void ascentPublishExecute(const conduit::Node& mesh);

    // Finalize the visualization and clean up resources
    void finalize();

    // Print statistics (rendering time, etc.)
    void printStats() const;

    // Run the visualization process at a specific time and timestep
    void run(double time, int tstep);

    // Set affinity for main and queue thread
    void setAffinity(pthread_t threadID);

    // Setup visualization with required data
    void setup(nrs_t *nrs_, int interval, std::string mode, std::string nekrsAffinity, std::string ascentAffinity);

    // Wait for all tasks to complete before proceeding
    void waitForCompletion();


  private:

    ascent::Ascent mAscent;    // Ascent instance for visualization
    std::string mAscentAffinity; // Affinity for Ascent
    std::queue<std::shared_ptr<SimulationData>> mDataQueue; // Queue of data for rendering
    double mElapsedAscentExecuteTime = 0.0; 
    double mElapsedAscentPublishTime = 0.0; 
    double mElapsedDataTransferTime = 0.0;
    bool mInitialized = false;
    int mInterval = -1;        // Track how often to run insitu
    mesh_t *mMesh;             // Pointer to the mesh object
    std::string mMode;         // Synchronous or asynchronous
    std::string mNekrsAffinity; // Affinity for Nekrs
    nrs_t *mNrs;               // Pointer to the solver object
    std::condition_variable mQueueCondition; // Condition variable for signaling   
    std::mutex mQueueMutex;    // Mutex to protect queue access
    bool mShouldStopThread = false;  // Flag to signal when to stop the thread
    std::thread mWorkerThread; // Single thread for rendering tasks

    // Internal helper functions
    size_t calculateTotalMemoryNeeded(dlong Nlocal) const;
    void copyDataToCPU();      // Copy simulation data to CPU (if needed)
    bool hasEnoughMemory(size_t requiredMemory) const;
    void initializeAscent();   // Initialize Ascent if needed
    std::vector<int> parseStringToVector(const std::string& input);
    void workerThreadFunc();   // Function that the worker thread runs
};

#endif // NEKRS_ASCENT_HPP