require 'torch'
require 'xlua'
require 'dok'
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
      'sfm.sba_expert',
      'Computes structure from motion using sba-1.6',
      {arg='motstruct', type='torch.Tensor', 
       help='matrix of nFrames x 3 motion parameters and nPts x 3 structure parameters', req=true},
      {arg="nframes", type='number',
       help='number of frames in set', req=true},
      {arg="n3Dpts", type='number',
       help='number of 3D points in set', req=true},
      {arg="vmask", type='torch.CharTensor',
       help='nframes x n2Dpts', req=true},
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

function sfm.sba(...)
   local _, projections, vmask, cameras, points3D, calibration = 
      xlua.unpack(
      {...},
      'sfm.sba',
      'Computes structure from motion using sba-1.6',
      {arg='projections', type='torch.Tensor', 
       help='projection of points back into frames n2Dpts x 2', req=true},
      {arg="vmask", type='torch.CharTensor',
       help='nframes x n2Dpts', req=true},
      {arg='cameras', type='torch.Tensor', 
       help='matrix of camera rotations and translation ncams x DOF'},
      {arg='points3D', type='torch.Tensor', 
       help='list of points npts x 3'},
      {arg='calibration', type='torch.Tensor',
       help='matrix of intrinsic camera parameters (3x3)'}
   )
      
end
function sfm.project3D(point,calibration)
   local point3D = torch.Tensor(3):fill(1)
   -- xd = (xp - cc1)/fc1 -alpha_c*yd
   -- yd = (yp - cc2)/fc2
   -- assume alpha_c is zero for now
   point3D[2] = 
      (point[2] - calibration[2][3] )/calibration[2][2]
   point3D[1] = 
      (point[1] - calibration[1][3])/calibration[1][1]
   return point3D
end

-- accepts points in a n3Dpoints x nframes x 2 (x,y) matrix, builds
-- the necessary data structures to pass this to sba returns the
-- motion, structure and calibration parameters computed
function sfm.sba_points (...)
   local _,projections,calibration = dok.unpack (
      {...},
      'sfm.sba_points',
      'creates all necessary structs for sparse bundle adjust, returns motion, structure(3D points) and calibration computed',
      {arg='projections', type='torch.Tensor | table', 
       help='n3Dpoints x nframes x 2', req=true},
      {arg='calibration', type='torch.Tensor',
       help='calibration matrix for camera (3x3)'}
   )
   local cnp = 6
   local mnp = 2
   local pnp = 3
   if not calibration then
      calibration=torch.Tensor(3,3):fill(0)
      -- make many assumptions
      -- from: http://phototour.cs.washington.edu/focal.html
      -- focal length in pixels = 
      --   (image width in pixels) * 
      --   (focal length in mm) / (CCD width in mm)
      -- cnp = cnp + 7
      local focalmm = 5.4
      local CCDmm   = 5.27
      local width   = 640
      local height  = 480
      local focalpx  = width * focalmm/CCDmm
      calibration[1][1] = focalpx
      calibration[2][2] = focalpx
      calibration[3][3] = 1
      calibration[1][3] = width/2
      calibration[2][3] = height/2
   end

   local n3Dpts  = 0
   local nframes = 0
   local n2Dproj = 0
   if type(projections) == 'table' then
      for i = 1,#projections do
         n3Dpts = n3Dpts + projections[i]:size(1)
         nframes = nframes + projections[i]:size(2)
         n2Dproj = projections[i]:size(1) * projections[i]:size(2)
      end
   else
      n3Dpts  = projections:size(1)
      nframes = projections:size(2)
      n2Dproj = n3Dpts * nframes
   end
   local vmask     = torch.CharTensor(n3Dpts,nframes):fill(0)
   local motstruct = torch.Tensor(nframes*cnp + n3Dpts*pnp):fill(0)
   local motion    = motstruct:narrow(1,1,nframes*cnp):resize(nframes,cnp)
   local structure = motstruct:narrow(1,nframes*cnp+1,n3Dpts*pnp):resize(n3Dpts,pnp)

   local fullquatz = 4
   local initrot   = torch.Tensor(nframes,fullquatz):fill(0)
   -- set initial rotation estimate to quaterion zero {1,0,0,0}
   initrot:select(2,1):fill(1)
   local imgprojs  = torch.Tensor(n2Dproj,mnp)
   local cproj = 1
   -- FIXME make this work for tables also
   for i = 1,n3Dpts do
      -- approximate 3D based on projection using first point
      local pt3d = sfm.project3D(projections[i][1],calibration)
      structure[i]:copy(pt3d)
      for j = 1,nframes do 
         imgprojs[j]:copy(projections[i][j])
         vmask[i][j] = 1
      end
   end
   print("calling sba_driver")
   motstruct.libsfm.sba_driver(motstruct,nframes,n3Dpts,vmask,
                               initrot,imgprojs, calibration)
   return motion,structure,calibration
end

function sfm.sba_testme () 
   -- similar to the expert case but the initrot and and vmask are
   -- computed int the sfm.sba() function
   local camerafname = sys.concat(sys.fpath(), "7cams.txt")
   local pointsfname = sys.concat(sys.fpath(), "7pts.txt")
   local calibfname  = sys.concat(sys.fpath(), "calib.txt")
   print("opening "..camerafname)
   local nframes=tonumber(sys.execute("wc -l "..camerafname):match("%d+"))
   sfm.cnp = 6
   sfm.pnp = 3
   sfm.mnp = 2
   local cnp = sfm.cnp
   local pnp = sfm.pnp
   local mnp = sfm.mnp
   local filecnp = 7
   local camera    = torch.Tensor(nframes,filecnp)
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
   -- it seems so f*cking broken to move back and forth between
   -- quaterion mode and non-quaterion
   for i = 1,nframes do
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

function sfm.sba_expert_testme () 
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
