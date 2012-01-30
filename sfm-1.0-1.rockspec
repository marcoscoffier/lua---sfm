
package = "sfm"
version = "1.0-1"

source = {
   url = "sfm-1.0-1.tgz"
}

description = {
   summary = "A package to compute sfm on image sequences",
   detailed = [[
Wraps sparse bundle adjustment from http://www.ics.forth.gr/~lourakis/sba+
   ]],
   homepage = "",
   license = "GNU GPL + CeCILL"
}

dependencies = {
   "lua >= 5.1",
   "torch",
   "sys",
   "xlua",
   "image"
}

build = {
   type = "cmake",

   variables = {
      CMAKE_INSTALL_PREFIX = "$(PREFIX)"
   }
}
