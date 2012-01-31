#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/sba.c"
#else

static int libsba_(driver) (lua_State *L) {
  return 0;
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libsba_(Main__) [] = 
{
  {"sba_driver",        libsba_(driver)},
  {NULL, NULL}  /* sentinel */
};

DLL_EXPORT int libsba_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libsba_(Main__), "libsba");
  return 1; 
}

#endif
