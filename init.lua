require 'torch'
require 'xlua'
require 'sys'
require 'image'

sfm = {}

-- c lib:
require 'libsfm'

sfm.cnp = 6 -- 3 rot params + 3 trans params 
sfm.pnp = 3 -- euclidean 3D points 
sfm.mnp = 2 -- x,y projection in image

-- this should be renamed expert
function sfm.sba (...)
   local _, motstruct, nframes, n3Dpts, vmask, initrot, 
   imgproj, calibration = xlua.unpack(
      {...},
      'sfm.sba',
      'Computes structure from motion using sba-1.6',
      {arg='motstruct', type='torch.Tensor', 
       help='matrix of nFrames x 3 motion parameters and nPts x 3 structure parameters', req=true},
      {arg="nframes", type='number',
       help='', req=true},
      {arg="n3Dpts", type='number',
       help='', req=true},
      {arg="vmask", type='torch.CharTensor',
       help='', req=true},
      {arg='initrot', type='torch.Tensor', 
       help='matrix of camera rotations ncams x 4', req=true},
      {arg='imgproj', type='torch.Tensor', 
       help='projection of points back into frames n3Dpts x 2', req=true},
      {arg='calibration', type='torch.Tensor', 
       help='matrix of intrinsic camera parameters'}
   )
   print("in sfm.sba()")
   motstruct.libsfm.sba_driver(motstruct,nframes,n3Dpts,vmask,
                               initrot,imgproj, calibration)
end

function sfm.sba_testme () 
   local camerafname = sys.concat(sys.fpath(), "7cams.txt")
   local pointsfname = sys.concat(sys.fpath(), "7pts.txt")
   print("opening "..camerafname)
   local nframes=tonumber(sys.execute("wc -l "..camerafname):match("%d+"))
   sfm.cnp = 6
   sfm.pnp = 3
   sfm.mnp = 2
   local cnp = sfm.cnp
   local pnp = sfm.pnp
   local mnp = sfm.mnp

   local camera = torch.Tensor(nframes*cnp)
   local camera_df = torch.DiskFile(camerafname)
   for i = 1,camera:nElement() do 
      camera[i] = camera_df:readDouble()
   end
   camera:resize(nframes,cnp)
   print(camera)
   camera_df:close()

   print("opening "..pointsfname)
   local n3Dpts=sys.execute("wc -l "..pointsfname):match("%d+")
   n3Dpts=n3Dpts-1
   local n2Dprojs = 0
   local points_df = torch.DiskFile(pointsfname)
   local header = points_df:readString("*l")
   print(header)
   for i = 1,n3Dpts do
      points_df:readDouble()
      points_df:readDouble()
      points_df:readDouble()
      n2Dprojs = n2Dprojs + points_df:readInt()
      -- read to end of line
      points_df:readString("*l")
   end
   print("n3Dpts: "..n3Dpts.." n2Dproj: "..n2Dprojs)
   local vmask     = torch.CharTensor(n3Dpts,nframes):fill(0)
   local imgpts    = torch.Tensor(n2Dprojs * mnp)
   local motstruct = torch.Tensor(nframes*cnp + n3Dpts*pnp)
   local motion = motstruct:narrow(1,1,nframes*cnp)
   motion:resize(nframes,cnp)
   local fullquatz = 4
   local initrot   = torch.Tensor(nframes,fullquatz)
   motion:copy(camera)
   for i = 1,camera:size(1) do 
      initrot[i][2] = camera[i][1+cnp-6]
      initrot[i][3] = camera[i][1+cnp-5]
      initrot[i][4] = camera[i][1+cnp-4]
      initrot[i][1] = math.sqrt(1 
                                - initrot[2]*initrot[2] 
                                - initrot[3]*initrot[3]
                                - initrot[4]*initrot[4])
   end
   -- now reread the points file
   points_df:seek(1)
   points_df:readString("*l") -- skip header 
   local structure = motstruct:narrow(1,nframes*cnp,n3Dpts*pnp)
   structure:resize(n3Dpts,pnp)
   local imgprojs = torch.Tensor(n2Dprojs,mnp)
   local cproj = 1
   for i = 1,n3Dpts do
      structure[i][1] = points_df:readDouble()
      structure[i][2] = points_df:readDouble()
      structure[i][3] = points_df:readDouble()
      local nframes = points_df:readInt()
      for j = 1,nframes do 
         local fnum = points_df:readInt()
         imgprojs[cproj][1] = points_df:readDouble()
         imgprojs[cproj][2] = points_df:readDouble()
         vmask[i][fnum+1]=1
      end
   end
   print("about to call sfm.sba()")
   sfm.sba(motstruct,nframes,n3Dpts,vmask,initrot,imgprojs,calibration)
end