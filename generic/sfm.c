#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/sfm.c"
#else

static int libsfm_(sba_driver) (lua_State *L) {
  return 0;
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
