#if !defined(nekrs_bctype_h_)
#define nekrs_bctype_h_

/*
    header file used in C++, Fortran and okl files

    boundary type IDs need to be index-1
    lower value have higher precedence  
*/

#define p_bcType_interpolation 1
#define p_bcType_zeroDirichlet 2
#define p_bcType_udfDirichlet 3

#define p_bcType_zeroDirichletX_zeroNeumann 4
#define p_bcType_zeroDirichletY_zeroNeumann 5
#define p_bcType_zeroDirichletZ_zeroNeumann 6
#define p_bcType_zeroDirichletN_zeroNeumann 7

#define p_bcType_zeroDirichletX_udfNeumann 8
#define p_bcType_zeroDirichletY_udfNeumann 9
#define p_bcType_zeroDirichletZ_udfNeumann 10
#define p_bcType_zeroDirichletN_udfNeumann 11

#define p_bcType_zeroDirichletYZ_zeroNeumann 12
#define p_bcType_zeroDirichletXZ_zeroNeumann 13
#define p_bcType_zeroDirichletXY_zeroNeumann 14
#define p_bcType_zeroDirichletT_zeroNeumann 15

#define p_bcType_udfRobin 16

#define p_bcType_zeroNeumann 17
#define p_bcType_udfNeumann 18

#define p_bcType_none 19 

#define p_NBcType 19

#endif
