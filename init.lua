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

-- takes an already created motstruct,vmask,initrot and imgproj
function sfm.sba_expert (...)
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
   motstruct.libsfm.sba_driver(motstruct,nframes,n3Dpts,vmask,
                               initrot,imgproj, calibration)
end

function sfm.sba_testme () 
   local camerafname = sys.concat(sys.fpath(), "7cams.txt")
   local pointsfname = sys.concat(sys.fpath(), "7pts.txt")
   local calibfname = sys.concat(sys.fpath(), "calib.txt")
   print("opening "..camerafname)
   local nframes=tonumber(sys.execute("wc -l "..camerafname):match("%d+"))
   sfm.cnp = 6
   sfm.pnp = 3
   sfm.mnp = 2
   local cnp = sfm.cnp
   local pnp = sfm.pnp
   local mnp = sfm.mnp
   local filecnp = 7
   local camera = torch.Tensor(nframes,filecnp)
   local camera_df = torch.DiskFile(camerafname)

   camera:copy(torch.Tensor(camera_df:readDouble(nframes*(filecnp))))
   camera_df:close()

   print("opening "..pointsfname)
   local n3Dpts=sys.execute("wc -l "..pointsfname):match("%d+")
   n3Dpts=n3Dpts-1
   local n2Dprojs = 0
   local points_df = torch.DiskFile(pointsfname)
   local header = points_df:readString("*l")
   for i = 1,n3Dpts do
      points_df:readDouble()
      points_df:readDouble()
      points_df:readDouble()
      n2Dprojs = n2Dprojs + points_df:readInt()
      -- read to end of line
      points_df:readString("*l")
   end
   local vmask     = torch.CharTensor(n3Dpts,nframes):fill(0)
   local imgpts    = torch.Tensor(n2Dprojs * mnp)
   local motstruct = torch.Tensor(nframes*cnp + n3Dpts*pnp)
   local motion = motstruct:narrow(1,1,nframes*cnp)
   motion:resize(nframes,cnp)
   local fullquatz = 4
   local initrot   = torch.Tensor(nframes,fullquatz):fill(0)
   -- go from quaternions in camera (7 params) to 3 rotation and 3
   -- position (6 params) in motion (quat2vec)

   -- it seems so f*cking broken to move back and forth between
   -- quaterion mode and non-quaterion

   for i = 1,nframes do
      local mag = math.sqrt(camera[i][1] * camera[i][1] +
                            camera[i][2] * camera[i][2] +
                            camera[i][3] * camera[i][3] +
                            camera[i][4] * camera[i][4])
      local sg = 1
      if camera[i][1] < 0 then sg = -1 end
      mag = sg/mag
      motion[i][1] = camera[i][2]*mag
      motion[i][2] = camera[i][3]*mag
      motion[i][3] = camera[i][4]*mag
      -- translation
      motion[i][4] = camera[i][5]
      motion[i][5] = camera[i][6]
      motion[i][6] = camera[i][7]
   end
   for i = 1,nframes do
      -- in quaternion mode the rotation parameters start in 2nd column
      initrot[i][2] = motion[i][1]
      initrot[i][3] = motion[i][2]
      initrot[i][4] = motion[i][3]
      initrot[i][1] = math.sqrt(1 
                                - initrot[i][2]*initrot[i][2] 
                                - initrot[i][3]*initrot[i][3]
                                - initrot[i][4]*initrot[i][4])
   end
   -- now reread the points file
   points_df:seek(1)
   points_df:readString("*l") -- skip header 
   local structure = motstruct:narrow(1,nframes*cnp+1,n3Dpts*pnp)
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
         cproj = cproj+1
      end
   end
   points_df:close()

   local calibration=torch.Tensor(3,3)
   local calib_df = torch.DiskFile(calibfname)
   print("opening "..calibfname)

   calibration:copy(torch.Tensor(calib_df:readDouble(9)))
   calib_df:close()
   
   sfm.sba_expert(motstruct,nframes,n3Dpts,vmask,initrot,imgprojs,calibration)
end