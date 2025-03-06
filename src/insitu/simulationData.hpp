#ifndef SIMULATION_DATA_HPP
#define SIMULATION_DATA_HPP

#include <variant>
#include <vector>
#include <mutex>

#include <ascent.hpp>

#include "nrs.hpp"


class SimulationData {
  private:
    // Define a template for 3D components (x, y, z)
    template <typename T>
      struct VectorComponents {
        std::variant<std::vector<T>, T*> x;  // Coordinates in x
        std::variant<std::vector<T>, T*> y;  // Coordinates in y
        std::variant<std::vector<T>, T*> z;  // Coordinates in z
      };

    struct Mesh : public VectorComponents<dfloat> {};
    struct Velocity : public VectorComponents<dfloat> {};

    using VectorOrPointer = std::variant<std::vector<dfloat>, dfloat*>;

    conduit::Node mConduitData;
    static std::vector<dlong> mConnectivity;   // Connectivity, does not change from timestep to timestep
    static std::once_flag mInitializationFlag;
    static Mesh mMesh; // Mesh, does not change from timestep tp timestep
    std::string mMode;
    dlong mNlocal;
    dlong mNSfields;
    dlong mNvertices;
    VectorOrPointer mPressure;       // Pressure data
    std::string mRuntime;
    std::unordered_map<std::string, VectorOrPointer> mScalars;  // Scalars (multiple fields)
    static std::vector<unsigned int> mGhosts;
    Velocity mVelocity;

    void calculateConnectivity(const mesh_t *mesh);
    void computeGhosts(const mesh_t *mesh);
    void constructConduitNode(const mesh_t *mesh, const nrs_t *nrs, const double time, const int tstep);

    template<typename MeshVariant>
      dfloat* getComponent(MeshVariant& mesh_variant);

    template<typename T>
      dfloat* getComponent(T& component, const std::string& axis);

    void setMesh(const mesh_t *mesh);
    void setPressure(const nrs_t *nrs);
    void setScalars(const nrs_t *nrs);
    void setVelocity(const nrs_t *nrs);

  public:
    SimulationData(mesh_t *mesh, nrs_t *nrs, std::string mode, const double time, const int tstep);

    const conduit::Node& getConduitData() const;
    dlong* getConnectivity();
    dfloat* getMesh(const std::string& axis);
    unsigned int* getGhosts();
    dfloat* getPressure();
    dfloat* getScalar(const std::string key);
    dfloat* getVelocity(const std::string& axis);

    // Generalized helper function for copying data
    template<typename VectorType, typename DeviceType>
      static void copy_to_variant(std::variant<VectorType, dfloat*>& variant, DeviceType& device_data, dlong N) {
        std::visit([&](auto& vec) {
            if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<dfloat>>) {
              vec.resize(N);
              device_data.copyTo(vec.data(), N);
              }
            }, variant);
      }


};

#endif