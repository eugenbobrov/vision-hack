#!/usr/bin/env bash
set -o nounset


mkdir train
for rawvideo in trainset/akn.*
do
video=${rawvideo%.left.avi}
video=${video/akn./}
video=${video/set/}
mkdir $video
ffmpeg -i $rawvideo -vf crop=960:720:960:0 $video.avi
ffmpeg -i $video.avi $video/%03d.${video/train\//}.png
rm $video.avi
done
