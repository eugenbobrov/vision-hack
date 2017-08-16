#!/usr/bin/env bash
set -o nounset


mkdir train
for rawvideo in trainset/akn.*
do
video=${rawvideo%.left.avi}
video=${video/akn./}
video=${video/set/}
ffmpeg -i $rawvideo -s 960x540 $video.avi
mkdir $video
ffmpeg -i $video.avi -vf crop=960:360:0:0 $video.mod.avi
rm $video.avi
ffmpeg -i $video.mod.avi $video/%03d.${video/train\//}.png
rm $video.mod.avi
done
