#ifndef comm_hpp_
#define comm_hpp_
class comm_t{
public:
  comm_t(MPI_Comm, MPI_Comm);

  enum class type{
    dfloat,
    dlong,
    hlong,
  };

  enum class op{
    sum,
    max,
    min,
  };

  std::string to_string() const {
    std::ostringstream ss;
    ss << "mpiRank = " << _mpiRank << std::endl;
    ss << "mpiCommSize = " << this->mpiCommSize() << std::endl;
    ss << "mpiCommLocalSize = " << this->mpiCommLocalSize() << std::endl;
    ss << "localRank = " << _mpiRankLocal << std::endl;
    return ss.str();
  }

  int allreduce(occa::memory recvbuf, int count, op op, MPI_Comm comm) const;
  
  MPI_Comm mpiCommParent() const { return _mpiCommParent; };
  int mpiRankParent() const { return _mpiRankParent; };
  int mpiCommParentSize() const
  { 
    int comm_size;  
    MPI_Comm_size(this->_mpiCommParent, &comm_size);
    return comm_size;
  };

  MPI_Comm mpiComm() const { return _mpiComm; };
  int mpiRank()const { return _mpiRank; };
  int mpiCommSize() const
  { 
    int comm_size;  
    MPI_Comm_size(this->_mpiComm, &comm_size);
    return comm_size;
  };

  MPI_Comm mpiCommLocal() const { return _mpiCommLocal; };
  int mpiRankLocal() const { return _mpiRankLocal; };
  int mpiCommLocalSize() const
  { 
    int comm_size;  
    MPI_Comm_size(this->_mpiCommLocal, &comm_size);
    return comm_size;
  };

  bool deviceEnabled() const { return useGPUAware; };

private:
  MPI_Comm _mpiCommParent;
  MPI_Comm _mpiComm;
  MPI_Comm _mpiCommLocal;

  int _mpiRankParent;
  int _mpiRank;
  int _mpiRankLocal;

  MPI_Datatype toMPI_Datatype(const occa::memory& t) const;
  MPI_Op toMPI_Op(op t) const;

  void reallocScratch(size_t Nbytes) const;
  bool useGPUAware;

  mutable occa::memory h_recvBuf;
  mutable occa::memory h_sendBuf;
  mutable void* recv;
  mutable void* send;

};

#endif
