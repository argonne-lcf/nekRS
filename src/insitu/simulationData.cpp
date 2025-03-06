#include "simulationData.hpp"

std::vector<dlong> SimulationData::mConnectivity;
SimulationData::Mesh SimulationData::mMesh;
std::vector<unsigned int> SimulationData::mGhosts;
std::once_flag SimulationData::mInitializationFlag;

SimulationData::SimulationData(mesh_t *mesh, nrs_t *nrs, std::string mode, const double time, const int tstep) 
  : mMode(mode), mNlocal(mesh->Nlocal) {

    // Initialize mConnectivity with the size calculated from mesh
    dlong num_elements = mesh->Nelements * (mesh->Nq - 1) * (mesh->Nq - 1) * (mesh->Nq - 1);
    mNvertices = num_elements * 8;
    mConnectivity.resize(mNvertices);

    std::call_once(mInitializationFlag, [this, mesh]() {
        setMesh(mesh);
        calculateConnectivity(mesh);
        //computeGhosts(mesh);
        });

    setVelocity(nrs);
    setPressure(nrs);
    setScalars(nrs);
    constructConduitNode(mesh, nrs, time, tstep);
  }

void SimulationData::calculateConnectivity(const mesh_t *mesh) {
  auto it = mConnectivity.begin();
  dlong Nq2 = mesh->Nq * mesh->Nq;  // Precompute Nq squared

  for(int elem=0; elem<mesh->Nelements; ++elem) {
    for(int z=0; z < mesh->Nq-1; ++z) {
      for(int y=0; y < mesh->Nq-1; ++y) {
        for(int x=0; x < mesh->Nq-1; ++x) {
          it[0] = ((elem * mesh->Nq + z) * mesh->Nq + y) * mesh->Nq + x;
          it[1] = it[0] + 1;
          it[2] = it[0] + mesh->Nq+1;
          it[3] = it[0] + mesh->Nq;
          it[4] = it[0] + Nq2;
          it[5] = it[1] + Nq2;
          it[6] = it[2] + Nq2;
          it[7] = it[3] + Nq2;
          it += 8; 
        }
      }
    }
  }
}

/*
void SimulationData::computeGhosts(const mesh_t *mesh) {
  // halo node FIXME: ??
  if( mesh->totalHaloPairs ) {
    for (int n=0; n<mesh->totalHaloPairs * mesh->Nfp; n++) {
      mGhosts[mesh->haloGetNodeIds[n]] = 1;
    }
  }
} */

void SimulationData::constructConduitNode(const mesh_t *mesh, const nrs_t *nrs, const double time, const int tstep) {
  cds_t *cds = nrs->cds;

  mConduitData["coordsets/coords/type"] = "explicit";
  mConduitData["coordsets/coords/values/x"].set_external(getMesh("x"), mesh->Nlocal);
  mConduitData["coordsets/coords/values/y"].set_external(getMesh("y"), mesh->Nlocal);
  mConduitData["coordsets/coords/values/z"].set_external(getMesh("z"), mesh->Nlocal);  

  mConduitData["topologies/mesh/type"]           = "unstructured";
  mConduitData["topologies/mesh/coordset"]       = "coords";
  mConduitData["topologies/mesh/elements/shape"] = "hex";
  mConduitData["topologies/mesh/elements/connectivity"].set_external(getConnectivity(), mNvertices);

  /*
  if( mesh->totalHaloPairs ) { // FIXME
    mConduitData["fields/ghosts/association"] = "vertex";
    mConduitData["fields/ghosts/topology"] = "mesh";
    mConduitData["fields/ghosts/values"].set_external(getGhosts(), mesh->Nlocal);
  }
  */

  
  mConduitData["fields/vel_x/association"]  = "vertex";
  mConduitData["fields/vel_x/topology"]     = "mesh";
  mConduitData["fields/vel_x/values"].set_external(getVelocity("x"), mesh->Nlocal);

  mConduitData["fields/vel_y/association"]  = "vertex";
  mConduitData["fields/vel_y/topology"]     = "mesh";
  mConduitData["fields/vel_y/values"].set_external(getVelocity("y"), mesh->Nlocal);

  mConduitData["fields/vel_z/association"]  = "vertex";
  mConduitData["fields/vel_z/topology"]     = "mesh";
  mConduitData["fields/vel_z/values"].set_external(getVelocity("z"), mesh->Nlocal);

  if(nrs->o_P.ptr()) {
    mConduitData["fields/pressure/association"]  = "vertex";
    mConduitData["fields/pressure/topology"]     = "mesh";
    mConduitData["fields/pressure/values"].set_external(getPressure(), mesh->Nlocal);
  }


  for (int is = 0; is < cds->NSfields; is++) {
    const std::string sid = (is==0) ? "temperature" : "scalar" + scalarDigitStr(is);
    mConduitData["fields/" + sid + "/association"]  = "vertex";
    mConduitData["fields/" + sid + "/topology"] = "mesh";
    mConduitData["fields/" + sid + "/values"].set_external(getScalar(sid), mesh->Nlocal);
  }

  mConduitData["state/cycle"] = tstep;
  mConduitData["state/time"] = time;
}

const conduit::Node& SimulationData::getConduitData() const {
  return mConduitData;
}

dlong* SimulationData::getConnectivity() {
  return mConnectivity.data();
}

template<typename MeshVariant>
inline dfloat* SimulationData::getComponent(MeshVariant& mesh_variant) {
  return std::visit([](auto&& arg) -> dfloat* {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, dfloat*>) {
      return arg;
      } else if constexpr (std::is_same_v<T, std::vector<dfloat>>) {
      return arg.data();
      }
      }, mesh_variant);
}

template<typename T>
dfloat* SimulationData::getComponent(T& component, const std::string& axis) {
  if (axis == "x") return getComponent(component.x);
  if (axis == "y") return getComponent(component.y);
  if (axis == "z") return getComponent(component.z);
  throw std::invalid_argument("Invalid axis specified. Use 'x', 'y', or 'z'.");
}

unsigned int* SimulationData::getGhosts() {
  return mGhosts.data();
}

dfloat* SimulationData::getMesh(const std::string& axis) {
  return getComponent(mMesh, axis);
}

dfloat* SimulationData::getPressure() {
  return getComponent(mPressure);
}

dfloat* SimulationData::getScalar(const std::string key) {
  return getComponent(mScalars[key]);
}

dfloat* SimulationData::getVelocity(const std::string& axis) {
  return getComponent(mVelocity, axis);
}

void SimulationData::setMesh(const mesh_t *mesh) {

  if(mMode == "synchronous") {
    mMesh.x = (dfloat*) mesh->o_x.ptr();
    mMesh.y = (dfloat*) mesh->o_y.ptr();
    mMesh.z = (dfloat*) mesh->o_z.ptr();
  } else {
    //mMesh.x = (dfloat *)calloc(mNlocal, sizeof(dfloat));
    //mMesh.y = (dfloat *)calloc(mNlocal, sizeof(dfloat));
    //mMesh.z = (dfloat *)calloc(mNlocal, sizeof(dfloat));

    //mesh->o_x.copyTo(mMesh.x, mNlocal * sizeof(dfloat));
    //mesh->o_y.copyTo(mMesh.y, mNlocal * sizeof(dfloat));
    //mesh->o_z.copyTo(mMesh.z, mNlocal * sizeof(dfloat));
    copy_to_variant(mMesh.x, mesh->o_x, mNlocal);
    copy_to_variant(mMesh.y, mesh->o_y, mNlocal);
    copy_to_variant(mMesh.z, mesh->o_z, mNlocal);
    //mMesh.x = std::vector<dfloat>(mesh_x, mesh_x + mNlocal);
    //mMesh.y = std::vector<dfloat>(mesh_y, mesh_y + mNlocal);
    //mMesh.z = std::vector<dfloat>(mesh_z, mesh_z + mNlocal);
  }
}

void SimulationData::setPressure(const nrs_t *nrs) {
  if(nrs->o_P.ptr()) {
    if(mMode == "synchronous") {
      mPressure = (dfloat*) nrs->o_P.ptr();
    } else {
      copy_to_variant(mPressure, nrs->o_P, mNlocal);
    }
  }
}

void SimulationData::setScalars(const nrs_t *nrs) {
  cds_t *cds = nrs->cds;

  for (int is = 0; is < cds->NSfields; is++) {
    const auto& sid = (is==0) ? "temperature" : "scalar" + scalarDigitStr(is);
    dlong Nlocal = (is) ? cds->meshV->Nlocal : cds->mesh[0]->Nlocal;
    auto o_scalar = cds->o_S + cds->fieldOffsetScan[is] * sizeof(dfloat);

    if(mMode == "synchronous") {
      mScalars[sid] = (dfloat*) o_scalar.ptr();
    } else {
      copy_to_variant(mScalars[sid], o_scalar, Nlocal);
    }
  }
}

void SimulationData::setVelocity(const nrs_t *nrs) {

  auto o_u = nrs->o_U + 0 * nrs->fieldOffset;
  auto o_v = nrs->o_U + 1 * nrs->fieldOffset;
  auto o_w = nrs->o_U + 2 * nrs->fieldOffset;

  if (mMode == "synchronous") {
    mVelocity.x = (dfloat*) o_u.ptr();
    mVelocity.y = (dfloat*) o_v.ptr();
    mVelocity.z = (dfloat*) o_w.ptr();
  } else {
    copy_to_variant(mVelocity.x, o_u, mNlocal);
    copy_to_variant(mVelocity.y, o_v, mNlocal);
    copy_to_variant(mVelocity.z, o_w, mNlocal);
  }
}