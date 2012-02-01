#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/sfm.c"
#else

static int libsfm_(sba_driver) (lua_State *L) {
  THTensor * motstruct   = luaT_checkudata(L,1, torch_(Tensor_id));
  int nframes            = luaL_checknumber(L,2);  
  int n3Dpts             = luaL_checknumber(L,3);  
  THCharTensor * vmask   = luaT_checkudata(L,4, torch_CharTensor_id);
  THTensor * initrot     = luaT_checkudata(L,5, torch_(Tensor_id));
  THTensor * imgproj     = luaT_checkudata(L,6, torch_(Tensor_id));
  THTensor * calibration = NULL;
  if (!lua_isnil(L,7)) {
    calibration = luaT_checkudata(L,7, torch_(Tensor_id));
  }
  /* FIXME get these from the sfm.cnp etc. */
  int cnp      = 6; /* 3 rot params + 3 trans params */
  int pnp      = 3; /* euclidean 3D points */
  int mnp      = 2;
  int numprojs = imgproj->size[0];
  double * motstruct_ptr;
  char * vmask_ptr;
  double * initrot_ptr;
  double * imgproj_ptr;
  double * calib_ptr;
  double * refcams_ptr;
  double * refpts_ptr;
  refcams_ptr=NULL;
  refpts_ptr=NULL;
  int i;
  
#ifdef TH_REAL_IS_DOUBLE
  /* avoid the copy if we can */
  motstruct_ptr = THDoubleTensor_data(motstruct);
  vmask_ptr     = THCharTensor_data(vmask);
  initrot_ptr   = THDoubleTensor_data(initrot);
  imgproj_ptr   = THDoubleTensor_data(imgproj);
  if ((calibration != NULL) && (calibration->size[1] > 0)) {
    calib_ptr  = THDoubleTensor_data(calibration); 
  }else{
    calib_ptr = NULL; 
  }
#else
  /* must copy into a double */
#endif /* TH_REAL_IS_DOUBLE */
  sba_driver_c(motstruct_ptr,nframes,n3Dpts,initrot_ptr,
               imgproj_ptr,numprojs,
               vmask_ptr,calib_ptr,
               cnp,pnp,mnp,
               quat2vec, vec2quat, cnp+1, refcams_ptr, refpts_ptr);  return 0;
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libsfm_(Main__) [] = 
{
  {"sba_driver",        libsfm_(sba_driver)},
  {NULL, NULL}  /* sentinel */
};

DLL_EXPORT int libsfm_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libsfm_(Main__), "libsfm");
  return 1; 
}

#endif
