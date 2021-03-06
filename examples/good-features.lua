require 'xlua'
require 'torch'
require 'opencv'
require 'sfm'

-- use profiler
p = xlua.Profiler()

p:start('loading images')
frames={}
frames[1] = image.load('frame-000025.png')
frames[2] = image.load('frame-000029.png')
-- frames[3] = image.load('frame-000027.png')
--frames[4] = image.load('frame-000031.png')
--frames[5] = image.load('frame-000033.png')
width = frames[1]:size(3)
height = frames[1]:size(2)
p:lap('loading images')

p:start('get features to track')
trackPointsP = opencv.GoodFeaturesToTrack{image=frames[1], count=1000}
nPoints = trackPointsP:size(1)
p:lap('get features to track')
nframes = #frames

dirtyPoints = torch.Tensor(nPoints,nframes,2):fill(0)
dirtyPoints:select(2,1):copy(trackPointsP)
cleanPoints = torch.Tensor(nPoints,nframes,2):fill(0)
cFlag = torch.CharTensor(nPoints):fill(1)
-- track triplet
for i = 2,nframes do
   p:start('track features')
   trackPoints = opencv.TrackPyrLK{pair={frames[i-1],frames[i]}, 
                                   points_in=trackPointsP}
   dirtyPoints:select(2,i):copy(trackPoints)
   p:lap('track features')

   p:start('backtrack features')
   backtrack= opencv.TrackPyrLK{pair={frames[i],frames[i-1]}, 
                                points_in=trackPoints}
   diff = trackPointsP - backtrack
   diff:pow(2)
   cdiff = diff:select(2,1) + diff:select(2,2)
   for j = 1,cdiff:size(1) do 
      if ((math.floor(cdiff[j]) > 1) or
          (trackPoints[j][1] < 1) or
          (trackPoints[j][2] < 1) or
          (trackPoints[j][1] > width) or
          (trackPoints[j][2] > height))
      then 
         cFlag[j] = 0
      end
   end
   p:lap('backtrack features')
   trackPointsP = trackPoints
end

p:start('prep clean points')
cIndex = 1
for i = 1,nPoints do 
   if (cFlag[i] == 1) then 
      cleanPoints[cIndex]:copy(dirtyPoints[i])
      cIndex = cIndex + 1
   end
end
cIndex = cIndex-1
cleanPoints = cleanPoints:narrow(1,1,cIndex)
print('found '..cIndex..' good points')
p:lap('prep clean points')

p:start('sparse bundle adjustment')
motion, structure, calibration = sfm.sba_points(cleanPoints)
p:lap('sparse bundle adjustment')

p:start('interpolate tracking')
flowfield = flowfield or 
   torch.Tensor(1, frames[nframes]:size(2), frames[nframes]:size(3))
depth = structure:narrow(2,2,1)
opencv.smoothVoronoi(cleanPoints:select(2,nframes), depth, flowfield)
p:lap('interpolate tracking')

p:start('display')
dispimg = frames[nframes]:clone()
opencv.drawFlowlinesOnImage{pair={cleanPoints:select(2,nframes),
                                  cleanPoints:select(2,nframes-1)}, 
                            image=dispimg}
dispimg:div(dispimg:max())

win = image.display{image={dispimg},win=win}
winf = image.display{image={flowfield},win=winf}
p:lap('display')

p:printAll()