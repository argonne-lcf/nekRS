#define bcDataField(a) (nekrs_strcmp(bc->fieldName, a) == 0)
#define bcDataFieldName(a) (nekrs_strcmp(bc->fieldName, a) == 0)
#define isField(a) (nekrs_strcmp(bc->fieldName, a) == 0)
#define isFieldName(a) (nekrs_strcmp(bc->fieldName, a) == 0)

#define strcmp nekrs_strcmp

inline char *_strcpy(char *destination, const char *source)
{
  char *original_dest = destination;
  while (*source != '\0') {
    *destination = *source;
    destination++;
    source++;
  }

  *destination = '\0';
  return original_dest;
}

inline char _toLower(char c)
{
  if (c >= 'A' && c <= 'Z') {
    return c + ('a' - 'A');
  }
  return c;
}

inline int nekrs_strcmp(const char *str1, const char *str2)
{
  while (*str1 != '\0' && *str2 != '\0' && (_toLower(*str1) == _toLower(*str2))) {
    str1++;
    str2++;
  }

  return _toLower(*str1) - _toLower(*str2);
}

struct bcData {
  char fieldName[32];

  // volume field index

  int idxVol;

  int fieldOffset;

  // boundary tag id
  int id;

  double time;

  // mesh coords
  dfloat x, y, z;

  // normal vector
  dfloat nx, ny, nz;

  // tangential vectors 
  dfloat t1x, t1y, t1z;
  dfloat t2x, t2y, t2z;

  // surface traction
  dfloat tr1, tr2;

  dfloat uxFluid, uyFluid, uzFluid;
  dfloat pFluid;
  dfloat uxFluidInt, uyFluidInt, uzFluidInt;

  int idScalar;
  dfloat sScalar;
  dfloat fluxScalar;
  dfloat sScalarInt;
  dfloat sInfScalar;
  dfloat h;
 
  dfloat uxGeom, uyGeom, uzGeom;

  dfloat transCoeff, diffCoeff;

  @globalPtr const dfloat *usrwrk;
};
