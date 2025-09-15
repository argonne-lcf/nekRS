#include "bcType.h"

c-----------------------------------------------------------------------
c
c NEK5000 Interface
c
c-----------------------------------------------------------------------
      subroutine nekf_bootstrap(comm_in,path_in,session_in,mesh_in)

      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
      include 'RESTART'
      include 'NEKINTF'

      integer comm_in
      character session_in*(*),path_in*(*)
      character mesh_in*(*)

      real rtest
      integer itest
      integer*8 itest8
      character ctest
      logical ltest 

      character*1  re2fle1(132)
      equivalence  (RE2FLE,re2fle1)

      ! set word size for REAL
      wdsize = sizeof(rtest)
      ! set word size for INTEGER
      isize = sizeof(itest)
      ! set word size for INTEGER*8
      isize8 = sizeof(itest8) 
      ! set word size for LOGICAL
      lsize = sizeof(ltest) 
      ! set word size for CHARACTER
      csize = sizeof(ctest)

      call setupcomm(comm_in,newcomm,newcommg,path_in,session_in)
      call iniproc()

      istep  = 0
      call initdim ! Initialize / set default values.
      call initdat
      call files

      call usrdat0 ! user may call nekrs_registerPtr
                   ! which we access in UDF_Setup0

      lp = 0 !ltrunc(PATH,132)
      call chcopy(re2fle1(lp+1),mesh_in,len(mesh_in))
      ls = lp + len(mesh_in)
      call blank(re2fle1(ls+1),len(re2fle)-ls)

      call nekrs_registerPtr('ndim', ndim)
      call nekrs_registerPtr('nelv', nelv)
      call nekrs_registerPtr('nelt', nelt)
      call nekrs_registerPtr('lelt', lelt)
      call nekrs_registerPtr('ldimt', ldimt)
      call nekrs_registerPtr('nx1', nx1)
      call nekrs_registerPtr('ifield', ifield)
      call nekrs_registerPtr('boundaryID', boundaryID)
      call nekrs_registerPtr('boundaryIDt', boundaryIDt)

      call nekrs_registerPtr('nekcomm', nekcomm)
      call nekrs_registerPtr('istep', istep)

      call nekrs_registerPtr('param', param)
      call nekrs_registerPtr('xc', xc)
      call nekrs_registerPtr('yc', yc)
      call nekrs_registerPtr('zc', zc)
      call nekrs_registerPtr('xm1', xm1)
      call nekrs_registerPtr('ym1', ym1)
      call nekrs_registerPtr('zm1', zm1)

      call nekrs_registerPtr('unx', unx)
      call nekrs_registerPtr('uny', uny)
      call nekrs_registerPtr('unz', unz)

      call nekrs_registerPtr('vx', vx)
      call nekrs_registerPtr('vy', vy)
      call nekrs_registerPtr('vz', vz)
      call nekrs_registerPtr('pr', pr)
      call nekrs_registerPtr('t', t)
      call nekrs_registerPtr('wx', wx)
      call nekrs_registerPtr('wy', wy)
      call nekrs_registerPtr('wz', wz)

      call nekrs_registerPtr('time', time)
      call nekrs_registerPtr('p0th', p0th)

      call nekrs_registerPtr('vmult', vmult)
      call nekrs_registerPtr('tmult', tmult)

      ! integer variants for (non-portable) booleans
      call nekrs_registerPtr('getxr', getxr)
      call nekrs_registerPtr('getur', getur)
      call nekrs_registerPtr('getpr', getpr)
      call nekrs_registerPtr('gettr', gettr)
      call nekrs_registerPtr('gtpsr', gtpsr(1))

      call nekrs_registerPtr('npsr', npsr)

      call nekrs_registerPtr('out_mask', out_mask)

      call nekrs_registerPtr('cbc', cbc)

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_setup(ifflow_in, 
     $                      bIDMap, bIDMapSize, bIDtMap, bIDtMapSize,
     $                      npscal_in, idpss_in, p32, mpart, contol,
     $                      rho, mue, rhoCp, lambda, stsform) 

      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
      include 'NEKINTF'

      integer bIDMapSize
      integer bIDtMapSize
      integer bIDMap(bIDMapSize)
      integer bIDtMap(bIDtMapSize)

      integer iftmsh_in, ifflow_in, mpart, p32
      integer idpss_in(*)
      real rho, mue, rhoCp, lambda, contol
      integer stsform

      common /rdump/ ntdump

      etimes = dnekclock_sync()

      call read_re2_hdr(ifbswap, .true.)

      if(ndim.eq.2) call exitti('Mesh has to be 3D!$', ndim) 

      call setDefaultParam

      loglevel   = 1
      cpfld(1,2) = rho
      cpfld(1,1) = mue
      cpfld(2,2) = rhoCp
      cpfld(2,1) = lambda

      param(27) = 1  ! torder 1 to save mem
      param(32) = p32 ! number of BC fields read from re2
      param(99) = -1 ! no dealiasing to save mem

      fluid_partitioner = mpart
      solid_partitioner = 1 ! fixed to RCB 
      connectivityTol = contol

      ifflow = .true.
      if(ifflow_in.eq.0) ifflow = .false.
      iftran = .true.
      ifheat = .false.
      ifvo   = .true.
      ifpo   = .true.
      if(stsform.eq.1) ifstrs = .true.

      if (npscal_in .gt. 0) then
        ifheat = .true.
        if(nelgt.ne.nelgv) iftmsh(2) = .true.
        if(nelgt.ne.nelgv .and. param(32).eq.1) param(32) = 2 
        npscal = npscal_in - 1
        param(23) = npscal
        ifto = .true.    
        call icopy(idpss, idpss_in, npscal+1)
      endif

      call usrdat0 ! call again just in case user want to change some params 

      call bcastParam
      call chkParam

      call mapelpr 
      call read_re2_data(ifbswap, .true., .true., .true.)

#if 0
      if(nid.eq.0) then
        write(6,*) 'bIDMap', bIDMap
        write(6,*) 'bIDtMap', bIDtMap
      endif
#endif

      ifld_bId = 2
      if(ifflow) ifld_bId = 1
      do iel = 1,nelv
      do ifc = 1,2*ndim
         boundaryID(ifc,iel) = -1
         if(bc(5,ifc,iel,ifld_bId).gt.0) then
           boundaryID(ifc,iel) = bc(5,ifc,iel,ifld_bId)
           idx = ibsearch(bIDMap, bIDMapSize, bc(5,ifc,iel,ifld_bId))
           if(idx.gt.0) then
             boundaryID(ifc,iel) = idx 
           endif
           bc(5,ifc,iel,ifld_bId) = boundaryID(ifc,iel)
         endif
      enddo
      enddo

      if(nelgt.ne.nelgv) then 
        do iel = 1,nelt
        do ifc = 1,2*ndim
         boundaryIDt(ifc,iel) = -1
         if(bc(5,ifc,iel,2).gt.0) then
           boundaryIDt(ifc,iel) = bc(5,ifc,iel,2)
           idx = ibsearch(bIDtMap, bIDtMapSize, bc(5,ifc,iel,2))
           if(idx.gt.0) then 
             boundaryIDt(ifc,iel) = idx
           endif
           bc(5,ifc,iel,2) = boundaryIDt(ifc,iel)
         endif
        enddo
        enddo
      endif

      call setvar          ! Initialize most variables

      igeom = 2
      call setup_topo      ! Setup domain topology

      if(.not. ifflow) then
        call rone(vmult,lx1*ly1*lz1*nelv)
        ifield = 1
        call dssum(vmult,lx1,ly1,lz1)
        call invcol1(vmult,lx1*ly1*lz1*nelv)
      endif

      call genwz           ! Compute GLL points, weights, etc.

      if(nio.eq.0) write(6,*) 'call usrdat'
      call usrdat
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat' 

      call gengeom(igeom)  ! Generate geometry, after usrdat 

      if(nio.eq.0) write(6,*) 'call usrdat2'
      call usrdat2
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat2' 

      call fix_geom
      call geom_reset(1)    ! recompute Jacobians, etc.

      call vrdsmsh          ! verify mesh topology

      call setlog(.false.)  ! Initalize logical flags

      call bcmask  ! Set BC masks for Dirichlet boundaries.

      ifield = 1

      if(nio.eq.0) write(6,*) 'call usrdat3'
      call usrdat3
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat3'

      time = 0.0
      p0thn = p0th
      ntdump = 0

      etimeSetup = dnekclock_sync() - etimes
      if(nio.eq.0) write(6,999) etimeSetup 
 999  format(' nek setup done in ', 1p1e13.4, ' s')
      if(nio.eq.0) write(6,*) 
      call flush(6)

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_outfld(fname, time_in, out_fld, 
     &                       nxo_in, rego_in,
     $                       xm_in, ym_in, zm_in,
     $                       vx_in, vy_in, vz_in,
     $                       pm1_in, t_in, ps_in, nps_in)

      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
      include 'NEKINTF'

      character fname*(*)
      real time_in
      integer out_fld(*)

      integer rego_in
      integer nxo_int

      real xm_in(*), ym_in(*), zm_in(*)
      real vx_in(*), vy_in(*), vz_in(*)
      real pm1_in(*)
      real t_in(*)
      real ps_in(lx1,ly1,lz1,lelt,*)
      integer nps_in

      integer*8 offs0,offs,nbyte,stride,strideB,nxyzo8
      logical rego

      integer cnt
      integer*8 cntg

      real time_s

      common /vrthov/ ur1(lxo*lxo*lxo*lelt)
     &              , ur2(lxo*lxo*lxo*lelt)
     &              , ur3(lxo*lxo*lxo*lelt)

      time_s = time
      time = time_in

      rego = .false.
      if(rego_in.ne.0) rego = .true.

      if(nxo_in.le.1) then
        nxo = nx1
      else
        nxo = nxo_in
      endif

      if (nxo.gt.lxo) then
        if (nid.eq.0) write(6,*) 
     &               'WARNING: nxo too large, reset to lxo!'
        nxo = lxo
      endif

      call io_init

      nout = nelt ! dump all fields based on the t-mesh to avoid different topologies in the post-processor
      nyo  = nxo 
      nzo  = nxo 

      ! open file
      ierr=0
      if (nid.eq.pid0) then
        !call mfo_open_files(prefix,ierr)
        !if(nio.eq.0)    write(6,*) '      FILE:',fname
        call byte_open_mpi(fname,ifh_mbyte,.false.,ierr)
      endif
      call err_chk(ierr,'Error opening file in mfo_open_files. $')

      call blank(rdcode1,10)
      i = 1
      if (out_fld(1).ne.0) then
         rdcode1(i)='X'
         i = i + 1
      endif
      if (out_fld(2).ne.0) then
         rdcode1(i)='U'
         i = i + 1
      endif
      if (out_fld(3).ne.0) then
         rdcode1(i)='P'
         i = i + 1
      endif
      if (out_fld(4).ne.0) then
         rdcode1(i)='T'
         i = i + 1
      endif
      if (nps_in.gt.0) then
         npscalo = 0
         do k = 1,nps_in
           if(out_fld(4+k).ne.0) npscalo = npscalo + 1
         enddo

         rdcode1(i) = 'S'
         write(rdcode1(i+1),'(I1)') npscalo/10
         write(rdcode1(i+2),'(I1)') npscalo-(npscalo/10)*10
      endif

      call mfo_write_hdr(rdcode1) ! hdr + element mapping

      cnt = 0
      do iel = 1,nelt
        if(out_mask(iel).ne.0) cnt = cnt + 1
      enddo
      cntg = iglsum(cnt, 1)

      nxyzo8  = nxo*nyo*nzo

      ! only relevant for single shared file
      offs0 = iHeaderSize + 4 + isize*cntg
      strideB = nelB * nxyzo8*wdsizo
      stride  = cntg * nxyzo8*wdsizo

      ioflds = 0
      if (out_fld(1).ne.0) then
         offs = offs0 + ldim*strideB
         call byte_set_view(offs,ifh_mbyte)
         call interp_fld_n(ur1,nxo,xm_in,rego)
         call interp_fld_n(ur2,nxo,ym_in,rego)
         call interp_fld_n(ur3,nxo,zm_in,rego)
         call mfo_outv(ur1,ur2,ur3,nout,nxo,nyo,nzo)
         ioflds = ioflds + ldim
      endif

      if (out_fld(2).ne.0) then
         offs = offs0 + ioflds*stride + ldim*strideB
         call byte_set_view(offs,ifh_mbyte)
         call interp_fld_n(ur1,nxo,vx_in,rego)
         call interp_fld_n(ur2,nxo,vy_in,rego)
         call interp_fld_n(ur3,nxo,vz_in,rego)
         call mfo_outv(ur1,ur2,ur3,nout,nxo,nyo,nzo)
         ioflds = ioflds + ldim
      endif

      if (out_fld(3).ne.0) then
         offs = offs0 + ioflds*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         call interp_fld_n(ur1,nxo,pm1_in,rego)
         call mfo_outs(ur1,nout,nxo,nyo,nzo)
         ioflds = ioflds + 1
      endif

      if (out_fld(4).ne.0) then
         offs = offs0 + ioflds*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         call interp_fld_n(ur1,nxo,t_in,rego)
         call mfo_outs(ur1,nout,nxo,nyo,nzo)
         ioflds = ioflds + 1
      endif

      do k=1,nps_in
         if (out_fld(4+k).ne.0) then
           offs = offs0 + ioflds*stride + strideB
           call byte_set_view(offs,ifh_mbyte)
           call interp_fld_n(ur1,nxo,ps_in(1,1,1,1,k),rego)
           call mfo_outs(ur1,nout,nxo,nyo,nzo)
           ioflds = ioflds + 1
         endif
      enddo
      dnbyte = 1.*ioflds*cnt*wdsizo*nxo*nyo*nzo

      ! add FP32 meta data (bounding boxes) to the end of the file
      if (if3d) then
         offs0   = offs0 + ioflds*stride
         strideB = nelB *2*4
         stride  = cntg *2*4
         ioflds  = 0
         if (out_fld(1).ne.0) then
            offs = offs0 + ldim*strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatav(xm1,ym1,zm1,nout)
            ioflds = ioflds + ldim
         endif
         if (out_fld(2).ne.0) then
            offs = offs0 + ioflds*stride + ldim*strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatav(vx_in,vy_in,vz_in,nout)
            ioflds = ioflds + ldim
         endif
         if (out_fld(3).ne.0) then
            offs = offs0 + ioflds*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatas(pm1_in,nout)
            ioflds = ioflds + 1
         endif
         if (out_fld(4).ne.0) then
            offs = offs0 + ioflds*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatas(t_in,nout)
            ioflds = ioflds + 1
         endif
         do k=1,nps_in
           if(out_fld(4+k).ne.0) then
             offs = offs0 + ioflds*stride + strideB
             call byte_set_view(offs,ifh_mbyte)
             call mfo_mdatas(ps_in(1,1,1,1,k), nout)
             ioflds = ioflds + 1
           endif
         enddo
         dnbyte = dnbyte + 2.*ioflds*cnt*wdsizo
      endif

      ierr = 0
      if (nid.eq.pid0) then 
         if(ifmpiio) then
           call byte_close_mpi(ifh_mbyte,ierr)
         else
           call byte_close(ierr)
         endif
      endif
      call err_chk(ierr,'Error closing file in mfo_outfld. Abort. $')

      time = time_s

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_restart(rfile,l)

      character*(l) rfile

      include 'SIZE'
      include 'RESTART'
      include 'INPUT'
      include 'NEKINTF'

      logical  iffort(  ldimt1,0:lpert)
     $       , ifrest(0:ldimt1,0:lpert)
     $       , ifprsl(  ldimt1,0:lpert)

      call blank(initc(1),132)
      call chcopy(initc(1),rfile,l)

      call slogic (iffort,ifrest,ifprsl,nfiles)

      call nekgsync()
      call restart(nfiles)
      call nekgsync()

      ! what fields exist in file
      getxr = 1
      if (.not. ifgetxr) getxr = 0 

      getur = 1
      if (.not. ifgetur) getur = 0 

      getpr = 1
      if (.not. ifgetpr) getpr = 0

      gettr = 1
      if (.not. ifgettr) gettr = 0

      do i = 1,ldimt-1
        gtpsr(i) = 1
        if (.not. ifgtpsr(i)) gtpsr(i) = 0
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_end()

      include 'mpif.h'
      include 'SIZE'
      include 'DPROCMAP'
      include 'RESTART'


#ifdef DPROCMAP
#ifdef MPI
      call MPI_Win_free(dProcmapH, ierr)
#endif
#endif 

#ifdef MPI
      if (commrs .ne. MPI_COMM_NULL) then
        call MPI_Win_free(rsH, ierr)
      endif
#endif

      return
      end
c-----------------------------------------------------------------------
      real function nekf_uf(u,v,w)

      real u(*), v(*), w(*)

      call nekuf(u,v,w)

      return
      end
c-----------------------------------------------------------------------
      integer function nekf_lglel(e)

      integer e

      include 'SIZE'
      include 'PARALLEL'

      nekf_lglel = lglel(e)

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_uic(ifld)

      include 'SIZE'
      include 'TSTEP'

      ifield_ = ifield
      ifield = ifld
      if (nio.eq.0) write(6,*) 'useric for ifld ', ifield
      call nekuic
      ifield = ifield_

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_ifoutfld(iswitch)

      include 'SIZE'
      include 'TSTEP'

      ifoutfld = .true.
      if (iswitch .eq. 0) ifoutfld = .false. 

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_setics()

      include 'SIZE'
      include 'RESTART'
      include 'NEKINTF'

      call setics()

      return
      end
c-----------------------------------------------------------------------
      integer function nekf_bcmap(bID, ifld, ismesh)

      include 'SIZE'
      include 'TOTAL'
      include 'NEKINTF'

      character*3 bcTypeT

      integer bID, ifld, ismesh
      character*3 c

      if (bID.lt.1) then ! not a boundary
        nekf_bcmap = 0
        return 
      endif 

      ibc = 0 
      c = cbc_bmap(bID, ifld)

      if (ifld.eq.1) then
        if (c.eq.'W  ') then 
          ibc = p_bcType_zeroDirichlet 
        else if (c.eq.'v  ') then 
          ibc = p_bcType_udfDirichlet 
          if(ismesh.eq.1) then
            ibc = p_bcType_zeroDirichlet 
          endif
        else if (c.eq.'int') then 
          ibc = p_bcType_interpolation
          if(ismesh.eq.1) then
            ibc = p_bcType_zeroDirichlet 
          endif
        else if (c.eq.'o  ' .or. c.eq.'O  ') then 
          ibc = p_bcType_zeroNeumann 
          if(ismesh.eq.1) then
            ! outflow remaps to SYM bounds for mesh solver
            ibc = p_bcType_zeroDirichletN_zeroNeumann 
          endif
        else if (c.eq.'on ' .or. c.eq.'ON ') then 
          ibc = p_bcType_zeroDirichletN_udfNeumann 
          if(ismesh.eq.1) then
            ! outflow remaps to SYM bounds for mesh solver
            ibc = p_bcType_zeroDirichletN_zeroNeumann 
          endif
        else if (c.eq.'onx') then 
          ibc = p_bcType_zeroDirichletYZ_zeroNeumann 
          if(ismesh.eq.1) then
            ! outflow remaps to SYM bounds for mesh solver
            ibc = p_bcType_zeroDirichletN_zeroNeumann 
          endif
        else if (c.eq.'ony') then 
          ibc = p_bcType_zeroDirichletXZ_zeroNeumann 
          if(ismesh.eq.1) then
            ! outflow remaps to SYM bounds for mesh solver
            ibc = p_bcType_zeroDirichletN_zeroNeumann 
          endif
        else if (c.eq.'onz') then 
          ibc = p_bcType_zeroDirichletXY_zeroNeumann 
          if(ismesh.eq.1) then
            ! outflow remaps to SYM bounds for mesh solver
            ibc = p_bcType_zeroDirichletN_zeroNeumann 
          endif
        else if (c.eq.'SYX') then 
          ibc = p_bcType_zeroDirichletX_zeroNeumann 
        else if (c.eq.'SYY') then 
          ibc = p_bcType_zeroDirichletY_zeroNeumann 
        else if (c.eq.'SYZ') then 
          ibc = p_bcType_zeroDirichletZ_zeroNeumann 
        else if (c.eq.'SYM') then 
          ibc = p_bcType_zeroDirichletN_zeroNeumann
        else if (c.eq.'shx') then 
          ibc = p_bcType_zeroDirichletX_udfNeumann 
        else if (c.eq.'shy') then 
          ibc = p_bcType_zeroDirichletY_udfNeumann 
        else if (c.eq.'shz') then 
          ibc = p_bcType_zeroDirichletZ_udfNeumann 
        else if (c.eq.'shl') then 
          ibc = p_bcType_zeroDirichletN_udfNeumann 
          if(ismesh.eq.1) then
            ! outflow remaps to SYM bounds for mesh solver
            ibc = p_bcType_zeroDirichletN_zeroNeumann 
          endif
         else if (c.eq.'mv ') then 
          ibc = p_bcType_udfDirichlet 
        endif
      else if(ifld.gt.1) then
        if (c.eq.'t  ') then 
          ibc = p_bcType_udfDirichlet 
        else if (c.eq.'int') then 
          ibc = p_bcType_interpolation 
        else if (c.eq.'o  ' .or. c.eq.'O  ' .or. c.eq.'I  ') then 
          ibc = p_bcType_zeroNeumann 
        else if (c.eq.'f  ') then 
          ibc = p_bcType_udfNeumann 
        else if (c.eq.'c  ') then 
          ibc = p_bcType_udfRobin 
        endif
      endif

      ibc_save = ibc
      ibc = iglmax(ibc, 1)

      if(nid.eq.0)
     $  write(6,*) ifld, 'bID: ', bID, 'cbc: ', 
     $             bcTypeT(ibc), ' bcTypeId: ', ibc

      if (ibc.eq.0) then
        write(6,*) 'Found unsupport BC type: ''', c , '''' 
        call exitt 
      endif

      if (ibc_save.gt.0 .and. ibc.ne.ibc_save) then
        write(6,*) 'Found ambiguous bcTypeId' 
        call exitt 
      endif

      nekf_bcmap = ibc

      return
      end
c-----------------------------------------------------------------------
      integer function nekf_nbid(isTmsh)

      include 'SIZE'
      include 'TOTAL'
      include 'NEKINTF'

      integer maxbid

      maxbid = 0
      if(isTmsh.eq.1) then
        n = 2*ndim*nelt
        do i = 1,n
           if(boundaryIDt(i,1) .gt. maxbid) maxbid = boundaryIDt(i,1) 
        enddo
      else
        n = 2*ndim*nelv
        do i = 1,n
           if(boundaryID(i,1) .gt. maxbid) maxbid = boundaryID(i,1) 
        enddo
      endif

      nekf_nbid = iglmax(maxbid,1)

      return
      end
c-----------------------------------------------------------------------
      integer*8 function nekf_set_vert(glo_num, nx, isTmsh, numberInt)

      include 'SIZE'
      include 'TOTAL'
      include 'NEKINTF'

      integer*8 glo_num(*)
      integer nx, isTmsh, numberInt

      common /ivrtx/ vertex ((2**ldim),lelt)
      integer*8 vertex

      integer*8 ngv
      logical numberInt_

      nel = nelt
      if (isTmsh.eq.0) nel = nelv
      numberInt_ = .false.
      if (numberInt.eq.1) numberInt_ = .true. 

      call set_vert(glo_num,ngv,nx,nel,vertex,numberInt_)

      nekf_set_vert = ngv

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_dssum(u)
      include 'SIZE'
      include 'TOTAL'

      ifld = ifield
      ifield = 1
      call dssum(u,lx1,ly1,lz1)
      ifield = ifld 

      return
      end
c-----------------------------------------------------------------------
      character*3 function bcTypeT(ibcType)

      bcTypeT = '???'

      if(ibcType.eq.p_bcType_zeroDirichlet) then 
         bcTypeT = 'W  '
      else if(ibcType.eq.p_bcType_interpolation) then
         bcTypeT = 'int'
      else if(ibcType.eq.p_bcType_udfDirichlet) then 
        bcTypeT = 'v  ' 
      else if(ibcType.eq.p_bcType_udfDirichlet) then 
        bcTypeT = 'mv ' 
      else if(ibcType.eq.p_bcType_zeroNeumann) then
        bcTypeT = 'O  ' 
      else if(ibcType.eq.p_bcType_zeroDirichletYZ_zeroNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletXZ_zeroNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletXY_zeroNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletN_udfNeumann) then
        bcTypeT = 'ON ' 
      else if(ibcType.eq.p_bcType_zeroDirichletX_zeroNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletY_zeroNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletZ_zeroNeumann .or.
     $        ibcType.eq.p_bcType_zeroDirichletN_zeroNeumann) then
        bcTypeT = 'SYM'
      else if(ibcType.eq.p_bcType_zeroDirichletX_udfNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletY_udfNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletZ_udfNeumann .or. 
     $        ibcType.eq.p_bcType_zeroDirichletN_udfNeumann) then
        bcTypeT = 'shl'
      else if(ibcType.eq.p_bcType_udfDirichlet) then
          bcTypeT = 't  ' 
      else if(ibcType.eq.p_bcType_zeroNeumann) then
          bcTypeT = 'I  ' 
      else if(ibcType.eq.p_bcType_udfNeumann) then 
          bcTypeT = 'f  '
      else if(ibcType.eq.p_bcType_udfRobin) then
          bcTypeT = 'c  ' 
      endif

      if (bcTypeT.eq.'???') then
          ierr = 1
          write(6,*) 'Found unsupport byTypeId: ''',  ibcType, '''' 
      endif

      end
c-----------------------------------------------------------------------
      integer function bcType(ifc,iel,ifld,ierr)

      include 'SIZE'
      include 'TOTAL'

      character*3 cb
      logical ifalg,ifnorx,ifnory,ifnorz

      ierr = 0
      cb = cbc(ifc,iel,ifld) 
      call chknord(ifalg,ifnorx,ifnory,ifnorz,ifc,iel)

      bcType = -1

      if(cb.eq.'W  ') then 
        bcType = p_bcType_zeroDirichlet
      else if(cb.eq.'int') then 
        bcType = p_bcType_interpolation
      else if(cb.eq.'v  ') then 
        bcType = p_bcType_udfDirichlet
      else if(cb.eq.'mv ') then 
        bcType = p_bcType_udfDirichlet 
      else if(cb.eq.'on ' .or. cb.eq.'ON ') then
        if (ifnorx) 
     $    bcType = p_bcType_zeroDirichletYZ_zeroNeumann 
        if (ifnory) 
     $    bcType = p_bcType_zeroDirichletXZ_zeroNeumann
        if (ifnorz) 
     $    bcType = p_bcType_zeroDirichletXY_zeroNeumann
        if (.not.ifalg) 
     $    bcType = p_bcType_zeroDirichletN_udfNeumann
      else if(cb.eq.'SYM') then
        if (ifnorx) 
     $    bcType = p_bcType_zeroDirichletX_zeroNeumann
        if (ifnory) 
     $    bcType = p_bcType_zeroDirichletY_zeroNeumann 
        if (ifnorz) 
     $    bcType = p_bcType_zeroDirichletZ_zeroNeumann 
        if (.not.ifalg) 
     $    bcType = p_bcType_zeroDirichletN_zeroNeumann
      else if(cb.eq.'shl') then
        if (ifnorx) 
     $    bcType = p_bcType_zeroDirichletX_udfNeumann 
        if (ifnory) 
     $    bcType = p_bcType_zeroDirichletY_udfNeumann 
        if (ifnorz) 
     $    bcType = p_bcType_zeroDirichletZ_udfNeumann 
        if (.not.ifalg) 
     $    bcType = p_bcType_zeroDirichletN_udfNeumann
      else if(cb.eq.'t  ') then 
          bcType = p_bcType_udfDirichlet
      else if(cb.eq.'I  ' .or. cb.eq.'O  ' .or. cb.eq.'o  ') then 
          bcType = p_bcType_zeroNeumann
      else if(cb.eq.'f  ') then 
          bcType = p_bcType_udfNeumann
      else if(cb.eq.'c  ') then 
          bcType = p_bcType_udfRobin
      endif

      if (bcType.eq.-1 .and. (cb.ne.'E  ' .and. cb.ne.'P  ')) then
          ierr = 1
          write(6,*) 'Found unsupport BC type: ''', cb , '''' 
      endif

      end
c-----------------------------------------------------------------------
      subroutine nekf_gen_bcmap()
c
c     generating cbc_bmap mapping a boundaryID to boundary type 
c
      include 'SIZE'
      include 'TOTAL'

      integer map2bID(p_NBcType)
      integer cnt(p_NBcType, ldimt1)

      integer bcType
      integer bID
      integer bIDcnt
      character*3 cb 

      call ifill(boundaryID, -1, size(boundaryID))
      call ifill(boundaryIDt,-1, size(boundaryIDt))

      if(.not.ifflow .and. .not.ifheat) return 

      call izero(cnt, size(cnt))

      ifldStart = 2
      if(ifflow) ifldStart = 1

      if(ifflow) then
        do iel = 1,nelv
        do ifc = 1,2*ndim
          ival = bcType(ifc,iel,1,ierr)
          if(ival.gt.0) cnt(ival, 1) = 1
          if(ierr.gt.0) goto 91 
        enddo
        enddo
 91     call err_chk(ierr, 'Invalid velocity boundary type!$')
      endif

      do ifld = 2,nfield
        ierr = 0
        if(idpss(ifld-1).lt.0 .or. iftmsh(ifld)) goto 50 

        do iel = 1,nelv
        do ifc = 1,2*ndim
          ival = bcType(ifc,iel,ifld,ierr) 
          if(ival.gt.0) cnt(ival, ifld) = 1
          if(ierr.gt.0) goto 92 
        enddo
        enddo

 50     continue
 92     call err_chk(ierr, 'Invalid scalar boundary type!$')
      enddo

      ! assign each bcType to a consecutive bID
      ifldMax = 0
      icntMax = 0 
      do ifld = ifldStart,nfield
        icnt = 0 
        do i = 1, p_NBcType
          cnt(i,ifld) = iglmax(cnt(i, ifld),1)
          if(cnt(i,ifld).gt.0) icnt = icnt + 1 
        enddo 
        if (icnt.gt.icntMax) then 
          ifldMax = ifld
          icntMax = icnt 
        endif 
      enddo

      call izero(map2bID, size(map2bID))
      bIDcnt = 0
      do i = 1,size(map2bID)
        if(cnt(i, ifldMax).gt.0) then
          bIDcnt = bIDcnt + 1
          map2bID(i) = bIDcnt
        endif 
      enddo

      do iel = 1,nelv
      do ifc = 1,2*ndim
        i = bcType(ifc,iel,ifldMax,ierr)
        if (i.gt.0) boundaryID(ifc,iel) = map2bID(i)
      enddo
      enddo

      do ifld = ifldStart,nfield
        ierr = 0
        if(idpss(ifld-1).lt.0 .or. iftmsh(ifld)) goto 51 

        do iel = 1,nelv
        do ifc = 1,2*ndim
          bID = boundaryID(ifc,iel)
          if (bID.gt.0) then
            if(cbc_bmap(bID,ifld).ne.'  ' .and. 
     $         cbc_bmap(bID,ifld).ne.cbc(ifc,iel,ifld)) then 
              ierr = 1
              goto 52
            endif
            cbc_bmap(bID,ifld) = cbc(ifc,iel,ifld)
          endif 
        enddo
        enddo

 51     continue
 52     call err_chk(ierr, 'Found ambiguous boundary id!$')
      enddo

      ! cht
      ifld = 2
      if(idpss(ifld-1).gt.-1 .and. iftmsh(ifld)) then
        call izero(cnt, size(cnt))

        ierr = 0
        do iel = 1,nelt
        do ifc = 1,2*ndim
          i = bcType(ifc,iel,ifld,ierr)
          if(i.gt.0) cnt(i, ifld) = 1
          if(ierr.gt.0) goto 93 
        enddo
        enddo

 93     call err_chk(ierr, 'Invalid temperature boundary type!$')

        do i = 1, p_NBcType
          cnt(i,ifld) = iglmax(cnt(i,ifld),1)
        enddo 

        call izero(map2bID, size(map2bID))
        bIDcnt = 0
        do i = 1,size(map2bID)
          if(cnt(i, ifld).gt.0) then
            bIDcnt = bIDcnt + 1
            map2bID(i) = bIDcnt
          endif 
        enddo
 
        do iel = 1,nelt
        do ifc = 1,2*ndim
          i = bcType(ifc,iel,ifldMax,ierr)
          if(i.gt.0) boundaryIDt(ifc,iel) = map2bID(i)
        enddo
        enddo

        do iel = 1,nelt
        do ifc = 1,2*ndim
          cbc_bmap(boundaryIDt(ifc,iel),ifld) = cbc(ifc,iel,ifld) 
        enddo
        enddo
      endif

      return
      end
c-----------------------------------------------------------------------
C
C     Generate geometric factors without updating coords
C
C----------------------------------------------------------------------
      subroutine nekf_updggeom()
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'GEOM'
      include 'WZ'

      COMMON /SCRUZ/ XM3 (LX3,LY3,LZ3,LELT)
     $ ,             YM3 (LX3,LY3,LZ3,LELT)
     $ ,             ZM3 (LX3,LY3,LZ3,LELT)

      ifld_save = ifield
      ifield = 1

      CALL LAGMASS
      call geom_reset(1)    ! recompute Jacobians, etc.

      ifield = ifld_save

      return
      end
c-----------------------------------------------------------------------
      subroutine nekrs_registerPtr(id, ptr)

      character id*(*)
      character ptr*(*)

      call nekf_registerPtr(id, ptr, len(id))

      return
      end
c-----------------------------------------------------------------------
      subroutine interp_fld_n(xn,n,xm,ifreg)   
c     Interpolate xm(m,m,m,...) to xn(n,n,n,...) (GLL-->GLL)

      include 'SIZE'
      include 'INPUT'

      real xn(*),xm(*)
      logical ifreg
      parameter (ldw=32*lx1*lx1*lz1)
      real work(ldw) ! Assumes size of map is < than single element
      integer im, in
      integer e

      m = nx1

      if (ifreg) then
        call map2reg(xn,n,xm,nelt)
        return
      endif

      if (n.eq.m) then
        call copy(xn,xm,m**ldim*nelt) 
        return
      endif

      mstride = m**ldim
      if (mstride.gt.ldw)
     $   call exitti('ABORT. ldw small in map_fld$',mstride)

      nstride = n**ldim
      if (nstride.gt.ldw)
     $   call exitti('ABORT. ldw small in map_fld$',nstride)

      im = 1
      in = 1
      do e = 1,nelt
         call map_m_to_n(xn(in),n,xm(im),m,if3d,work,ldw)
         im = im + mstride
         in = in + nstride
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_openfld(fname_in, time_, p0th_)
      include 'mpif.h'
      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
      include 'NEKINTF'

      character*(*) fname_in 
      real time_
      real p0th_

      integer nps_

      character*132  fname
      character*1    fnam1(132)
      equivalence   (fnam1,fname)

      character*1    frontc

      ifile = 1 ! single file only
      lenf = len(fname_in)

      ! add full path if required
      call blank(fname,132)
      call chcopy(frontc, fname_in, 1)

      if (frontc .ne. '/') then
        lenp = 0 !ltrunc(path,132)
        call chcopy(fnam1(1),path,lenp)
        call chcopy(fnam1(lenp+1),fname_in,lenf)
      else
        call chcopy(fnam1(1),fname_in,lenf)     
      endif

      call mfi_prepare(fname)       ! determine reader nodes +
                                    ! read hdr + element mapping 

      time_ = timer
      p0th_ = p0th

      ! what fields exist in file
      getxr = 1
      if (.not. ifgetxr) getxr = 0 

      getur = 1
      if (.not. ifgetur) getur = 0 

      getpr = 1
      if (.not. ifgetpr) getpr = 0

      gettr = 1
      if (.not. ifgettr) gettr = 0

      do i = 1,ldimt-1
        gtpsr(i) = 1
        if (.not. ifgtpsr(i)) gtpsr(i) = 0
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine nekf_readfld(xm1_,ym1_,zm1_,vx_,vy_,vz_,pm1_,t_,ps_)

      include 'mpif.h'
      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
      include 'NEKINTF'

      real xm1_(lx1,ly1,lz1,*), ym1_(lx1,ly1,lz1,*), zm1_(lx1,ly1,lz1,*)
      real vx_ (lx1,ly1,lz1,*), vy_ (lx1,ly1,lz1,*), vz_ (lx1,ly1,lz1,*)
      real pm1_(lx1,ly1,lz1,*)
      real t_  (lx1,ly1,lz1,*)
      real ps_ (lx1,ly1,lz1,lelt,*)

      parameter (lwk = 7*lx1*ly1*lz1*lelt)
      common /scrns/ wk(lwk)

      integer*8 offs0,offs,nbyte,stride,strideB,nxyzr8

      integer   disp_unit
      integer*8 win_size

#ifdef MPI
      disp_unit = 4 
      win_size  = int(disp_unit,8)*size(wk)
      if (commrs .eq. MPI_COMM_NULL) then
        call mpi_comm_dup(nekcomm,commrs,ierr)
        call MPI_Win_create(wk,
     $                      win_size,
     $                      disp_unit,
     $                      MPI_INFO_NULL,
     $                      commrs,rsH,ierr)

        if (ierr .ne. 0 ) call exitti('MPI_Win_allocate failed!$',0)
      endif
#endif

      offs0   = iHeadersize + 4 + isize*nelgr
      nxyzr8  = nxr*nyr*nzr
      strideB = nelBr* nxyzr8*wdsizr
      stride  = nelgr* nxyzr8*wdsizr

      iofldsr = 0
      if (ifgetxr) then
         offs = offs0 + ldim*strideB
         call byte_set_view(offs,ifh_mbyte)
         call mfi_getv(xm1_,ym1_,zm1_,wk,lwk,.false.) 
         iofldsr = iofldsr + ldim
      endif

      if (ifgetur) then
         offs = offs0 + iofldsr*stride + ldim*strideB
         call byte_set_view(offs,ifh_mbyte)
         call mfi_getv(vx_,vy_,vz_,wk,lwk,.false.)
         iofldsr = iofldsr + ldim
      endif

      if (ifgetpr) then
         offs = offs0 + iofldsr*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         call mfi_gets(pm1_,wk,lwk,.false.)
         iofldsr = iofldsr + 1
      endif

      if (ifgettr) then
         offs = offs0 + iofldsr*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         call mfi_gets(t_,wk,lwk,.false.)
         iofldsr = iofldsr + 1
      endif

      ierr = 0
      do k=1,npsr
          offs = offs0 + iofldsr*stride + strideB
          call byte_set_view(offs,ifh_mbyte)
          call mfi_gets(ps_(1,1,1,1,k),wk,lwk,.false.)
          iofldsr = iofldsr + 1
      enddo

      if(ifmpiio) then
        if(nid.eq.pid0r) call byte_close_mpi(ifh_mbyte,ierr)
      else
        if(nid.eq.pid0r) call byte_close(ierr)
      endif
      call err_chk(ierr,'Error closing restart file, in mfi.$')

      return
      end
