#include "crs_box.hpp"
#include "crs_xxt.hpp"

struct elem_t {
  ulong eid, vid[8];
  double xyz[8 * 3], mat[8 * 8], mask[8];
  int frontier[8];
  uint p, seq;
};

struct eid_t {
  ulong eid;
  uint e;
};

static int binary_search(ulong eid, struct eid_t *pe, uint n) {
  if (n == 0) return -1;

  uint l = 0, u = n - 1;
  while (u - l > 1) {
    uint mid = (u + l) / 2;
    if (pe[mid].eid == eid)
      return mid;
    else if (pe[mid].eid < eid)
      l = mid;
    else
      u = mid;
  }

  if (pe[l].eid == eid) return l;
  if (pe[u].eid == eid) return u;
  return -1;
}

static void fetch_nbrs_v3(uint *ne_, slong *vids, double *mat, uint nv,
                          uint ndim, uint nw, uint max_ne, MPI_Comm comm,
                          buffer *bfr) {
  // 1. Find neighbor elements of input elements based on vertex connectivity.
  struct vtx_t {
    ulong vid;
    ulong eid, nid;
    uint seq;
    uint p, np;
  };

  const size_t ne = *ne_;
  struct array vtxs;
  array_init(struct vtx_t, &vtxs, ne * nv);

  struct comm c;
  comm_init(&c, comm);
  struct crystal cr;
  crystal_init(&cr, &c);

  struct vtx_t vt = {.np = c.id};
  for (uint e = 0; e < ne; e++) {
    for (uint v = 0; v < nv; v++) {
      vt.vid = vids[e * nv + v], vt.p = vt.vid % c.np;
      array_cat(struct vtx_t, &vtxs, &vt, 1);
    }
  }

  sarray_transfer(struct vtx_t, &vtxs, p, 1, &cr);
  sarray_sort(struct vtx_t, vtxs.ptr, vtxs.n, vid, 1, bfr);

  struct array vtx2e;
  array_init(struct vtx_t, &vtx2e, vtxs.n);
  struct vtx_t *pv = (struct vtx_t *)vtxs.ptr;
  uint s = 0, e;
  while (s < vtxs.n) {
    e = s + 1;
    while (e < vtxs.n && pv[s].vid == pv[e].vid) e++;
    for (uint i = s; i < e; i++) {
      vt = pv[i];
      for (uint j = s; j < e; j++) {
        vt.np = pv[j].p, vt.nid = pv[j].eid;
        array_cat(struct vtx_t, &vtx2e, &vt, 1);
      }
    }
    s = e;
  }
  array_free(&vtxs);

  sarray_transfer(struct vtx_t, &vtx2e, p, 0, &cr);
  sarray_sort_2(struct vtx_t, vtx2e.ptr, vtx2e.n, seq, 0, nid, 1, bfr);

  // 2. Build element to neighbor map and element to processor map for input
  // elements.
  uint max_nbrs = 100 * max_ne;
  uint *offs = tcalloc(uint, max_ne + 1);
  ulong *nbrs = tcalloc(ulong, max_nbrs);
  uint *proc = tcalloc(uint, max_nbrs);

  ulong *elist = tcalloc(ulong, max_ne);
  uint *wlist = tcalloc(uint, max_ne);
  uint *plist = tcalloc(uint, max_ne);

  pv = (struct vtx_t *)vtx2e.ptr;
  s = 0, offs[0] = 0;
  uint cnt = 0;
  while (s < vtx2e.n) {
    elist[cnt] = pv[s].eid, wlist[cnt] = 0, plist[cnt] = c.id;

    e = s + 1;
    while (e < vtx2e.n && pv[s].eid == pv[e].eid) e++;

    // Check if `max_nbrs` is large enough.
    uint s0 = offs[cnt];
    if ((s0 + e - s) > max_nbrs) {
      fprintf(stderr, "Try max_nbrs larger than %d\n", s0 + e - s);
      fflush(stderr);
      exit(EXIT_FAILURE);
    }

    nbrs[s0] = pv[s].nid, proc[s0] = pv[s].np, s0++;
    for (uint i = s + 1; i < e; i++) {
      if (nbrs[s0 - 1] != pv[i].nid)
        nbrs[s0] = pv[i].nid, proc[s0] = pv[i].np, s0++;
    }
    cnt++, offs[cnt] = s0, s = e;
  }
  // Sanity checks.
  assert(cnt == ne);
  array_free(&vtx2e);

  // 3. Put all local elements in frontier array and sort by element id.
  // We will keep updating this and the map as we update the frontier.
  struct array fronta;
  array_init(struct eid_t, &fronta, (3 * ne) / 2);

  struct eid_t et;
  for (uint e = 0; e < ne; e++) {
    et.e = e;
    array_cat(struct eid_t, &fronta, &et, 1);
  }
  sarray_sort(struct eid_t, (eid_t *)fronta.ptr, fronta.n, eid, 1, bfr);

  struct array inputa;
  array_init(struct eid_t, &inputa, ne);
  array_cat(struct eid_t, &inputa, (eid_t *)fronta.ptr, fronta.n);

  // 4. Update the frontier by finding new neighbor elements from the previous
  // frontier.
  struct req_t {
    ulong eid;
    uint p, seq;
  };

  struct res_t {
    ulong eid, nid;
    uint p, np;
  };

  struct array rqsts;
  array_init(struct req_t, &rqsts, ne);

  struct array respns;
  array_init(struct res_t, &respns, rqsts.n * 10);

  uint fs = 0, fe = ne;
  for (uint w = 1; w <= nw; w++) {
    // Find all the new elements appearing in the map in last wave.
    for (uint i = fs; i < fe; i++) {
      for (uint s = offs[i], e = offs[i + 1]; s < e; s++) {
        if (binary_search(nbrs[s], (eid_t *)fronta.ptr, fronta.n) == -1) {
          struct eid_t et = {.eid = nbrs[s]};
          array_cat(struct eid_t, &fronta, &et, 1);
          // FIXME: This is bad.
          sarray_sort(struct eid_t, fronta.ptr, fronta.n, eid, 1, bfr);
          struct req_t rt = {.eid = nbrs[s], .p = proc[s]};
          array_cat(struct req_t, &rqsts, &rt, 1);
        }
      }
    }

    // Get the neighbors of the new elements.
    sarray_transfer(struct req_t, &rqsts, p, 1, &cr);
    sarray_sort(struct req_t, rqsts.ptr, rqsts.n, eid, 1, bfr);

    struct req_t *pr = (struct req_t *)rqsts.ptr;
    for (uint i = 0; i < rqsts.n; i++) {
      int idx = binary_search(pr[i].eid, (struct eid_t *)inputa.ptr, inputa.n);
      if (idx < 0 || idx >= ne) {
        fprintf(stderr, "Couldn't find element: %lld on processor: %d.",
                pr[i].eid, c.id);
        fflush(stderr);
        exit(EXIT_FAILURE);
      }

      struct res_t rt = {.eid = pr[i].eid, .p = pr[i].p};
      for (uint s = offs[idx], e = offs[idx + 1]; s < e; s++) {
        rt.nid = nbrs[s], rt.np = proc[s];
        array_cat(struct res_t, &respns, &rt, 1);
      }
    }

    sarray_transfer(struct res_t, &respns, p, 1, &cr);
    sarray_sort_2(struct res_t, respns.ptr, respns.n, eid, 1, nid, 1, bfr);

    // Update the map with the new elements and their neighbors.
    struct res_t *prs = (struct res_t *)respns.ptr;
    fs = fe, s = 0;
    while (s < respns.n) {
      if (fe >= max_ne) {
        fprintf(stderr, "max_ne: %u is too small. Try max_ne > %u", max_ne, fe);
        fflush(stderr);
        exit(EXIT_FAILURE);
      }

      elist[fe] = prs[s].eid, plist[fe] = prs[s].p, wlist[fe] = w, fe++;
      e = s + 1;
      while (e < respns.n && prs[s].eid == prs[e].eid) e++;

      offs[fe] = offs[fe - 1] + e - s;
      if (max_nbrs < offs[fe]) {
        fprintf(stderr, "max_nbrs: %u is too small. Try max_nbrs > %u",
                max_nbrs, offs[fe]);
        fflush(stderr);
        exit(EXIT_FAILURE);
      }

      for (uint i = 0; i < e - s; i++) {
        proc[offs[fe - 1] + i] = prs[s + i].np;
        nbrs[offs[fe - 1] + i] = prs[s + i].nid;
      }
      s = e;
    }
    rqsts.n = respns.n = 0;
  }
  array_free(&respns), array_free(&fronta), array_free(&inputa);
  free(offs), free(proc), free(nbrs);

  // 5. Now we have the element ids of the extended domain. We need to bring
  // other data in now. First we will put input data into an array and then
  // sort by element id (it could be that they are not sorted in input).
  struct array original;
  array_init(struct elem_t, &original, ne);

  struct elem_t elmt;
  for (uint i = 0; i < ne; i++) {
    elmt.p = c.id;
    for (uint v = 0; v < nv; v++) {
      elmt.vid[v] = vids[i * nv + v];
      for (uint j = 0; j < nv; j++)
        elmt.mat[v * nv + j] = mat[i * nv * nv + v * nv + j];
    }
    array_cat(struct elem_t, &original, &elmt, 1);
  }
  sarray_sort(struct elem_t, original.ptr, original.n, eid, 1, bfr);

  // 6. Now we are sending the requests to bring in the data for the extended
  // domain. Code doesn't distinguish between original and extended element
  // ids. It just asks for all the elemetns and sort them by element id.
  for (uint i = 0; i < fe; i++) {
    struct req_t rt = {.eid = elist[i], .p = plist[i], .seq = i};
    array_cat(struct req_t, &rqsts, &rt, 1);
  }
  assert(rqsts.n == fe);

  sarray_transfer(struct req_t, &rqsts, p, 1, &cr);
  sarray_sort(struct req_t, rqsts.ptr, rqsts.n, eid, 1, bfr);

  struct array extended;
  array_init(struct elem_t, &extended, rqsts.n);

  struct req_t *pr = (struct req_t *)rqsts.ptr;
  struct elem_t *po = (struct elem_t *)original.ptr;
  for (uint i = 0, j = 0; i < rqsts.n; i++) {
    while (j < ne && po[j].eid < pr[i].eid) j++;
    // Sanity check.
    assert(j < ne && po[j].eid == pr[i].eid);

    elmt.eid = po[j].eid, elmt.p = pr[i].p, elmt.seq = pr[i].seq;
    for (uint v = 0; v < nv; v++) {
      elmt.vid[v] = po[j].vid[v];
      for (uint k = 0; k < nv; k++)
        elmt.mat[v * nv + k] = po[j].mat[v * nv + k];
    }
    array_cat(struct elem_t, &extended, &elmt, 1);
  }
  array_free(&rqsts), array_free(&original);

  sarray_transfer(struct elem_t, &extended, p, 0, &cr);

  // 7. We have all the data now. Let's sort them by original element ids and
  // set the Fortran array correctly.
  sarray_sort(struct elem_t, extended.ptr, extended.n, seq, 0, bfr);
  struct elem_t *pe = (struct elem_t *)extended.ptr;
  *ne_ = fe;
  assert(fe < max_ne);
  for (uint i = 0; i < fe; i++) {
    // Sanity check.
    assert(elist[i] == pe[i].eid);
    for (uint v = 0; v < nv; v++) {
      vids[i * nv + v] = pe[i].vid[v];
      for (uint j = 0; j < nv; j++)
        mat[i * nv * nv + v * nv + j] = pe[i].mat[v * nv + j];
    }
  }
  array_free(&extended);
  free(elist), free(plist);

  // 8. Setup the Dirichlet BCs.
  uint ndof = fe * nv;
  sint *front = tcalloc(sint, ndof);
  for (uint e = 0; e < fe; e++) {
    if (wlist[e] == nw)
      for (uint v = 0; v < nv; v++) front[e * nv + v] = 1;
    else
      for (uint v = 0; v < nv; v++) front[e * nv + v] = 0;
  }

  struct comm lc;
  comm_split(&c, c.id, c.id, &lc);
  struct gs_data *gsh = gs_setup(vids, ndof, &lc, 0, gs_pairwise, 0);
  gs(front, gs_int, gs_min, 0, gsh, bfr);
  gs_free(gsh), comm_free(&lc);

  uint null_space = 1;
  for (uint i = 0; i < ndof; i++)
    if (front[i] == 1) null_space = vids[i] = 0;
  assert(null_space == 0);
  free(wlist), free(front);

  crystal_free(&cr), comm_free(&c);
}

void crs_box_setup_asm1(struct box *box) {
  const struct comm *const c = &(box->c);
  const uint nv = box->ncr;
  const uint nd = (nv == 8) ? 3 : 2;
  const uint nw = box->opts.nw;

  uint sne = box->un / nv;
  uint mne = 5 * sne + 200;
  slong *vtx = tcalloc(slong, nv * mne);
  double *va = tcalloc(double, nv * nv * mne);
  fetch_nbrs_v3(&sne, vtx, va, nv, nd, nw, mne, c->c, &box->bfr);

  box->sn = sne * nv;

  const uint nnz = box->sn * nv;
  uint *ia = tcalloc(uint, nnz);
  uint *ja = tcalloc(uint, nnz);
  for (uint e = 0; e < sne; e++) {
    for (uint j = 0; j < nv; j++) {
      for (uint i = 0; i < nv; i++) {
        ia[e * nv * nv + j * nv + i] = e * nv + i;
        ja[e * nv * nv + j * nv + i] = e * nv + j;
      }
    }
  }

  // Setup an isolated comm for each MPI process.
  struct comm lc;
  MPI_Comm local;
  MPI_Comm_split(c->c, c->id, c->id, &local);
  comm_init(&lc, local);
  MPI_Comm_free(&local);

  // Setup the ASM1 solver.
  ulong *ids = tcalloc(ulong, box->sn);
  for (uint i = 0; i < box->sn; i++) ids[i] = vtx[i];
  box->asm1 = (void *)crs_xxt_setup(box->sn, ids, nnz, ia, ja, va,
                                    box->opts.dom, 0 /* null space */, &lc);
  free(ids);
  comm_free(&lc);

  // Setup the crs_dsavg which basically average the solution of original
  // domains.
  for (uint i = box->un; i < box->sn; i++) vtx[i] = -vtx[i];
  box->gsh = gs_setup((const slong *)vtx, box->sn, c, 0, gs_auto, 0);

  free(vtx), free(ia), free(ja), free(va);
}
