from __future__ import division
from __future__ import print_function

import cv2
import pickle
import numpy as np
from pyannote.video import Video
import argparse
import os
from os import path
from PIL import Image
from Track import Tracks

from img2vec_pytorch import Img2Vec

target_shape = (225, 225)


def get_tracks_labels(labels_file):
    separator = ' '
    labels = {}
    if labels_file is not "":
        with open(labels_file) as f:
            for lines in f:
                line = lines.strip().split(separator)
                labels[line[0]] = line[1]
    print('track unique labels: ', set(list(labels.values())))
    return labels


def get_image(filename):
    return Image.open(filename)


def get_face_crop(frame, bb, frame_height, frame_width):
    (left, top, right, bottom) = bb
    left = int(float(left) * frame_width)
    right = int(float(right) * frame_width)
    top = int(float(top) * frame_height)
    bottom = int(float(bottom) * frame_height)

    area = (left, top, right, bottom)
    image = Image.fromarray(frame, mode='RGB')
    image = image.crop(area)
    image = np.array(image)
    return cv2.resize(image, dsize=target_shape, interpolation=cv2.INTER_CUBIC)


def get_tracks_to_skip(labels, track_ids):
    if len(labels) == 0:
        return []

    skip = []
    missing = []
    for tid in track_ids:
        if tid not in labels:
            missing.append(tid)
        if tid not in labels or labels[tid].lower() == "false_alarm":
            skip.append(tid)
    print('missing tids in labels file:', missing)
    return skip


def read_tracks(tracks_file, labels):
    separator = " "
    faces_by_time = {}
    with open(tracks_file) as f:
        lines = [line.split(separator) for line in f]
        track_ids = list(set([line[1] for line in lines]))
        skipping_track_ids = get_tracks_to_skip(labels, track_ids)
        print('skipping track ids: ', skipping_track_ids)
        lines = [line for line in lines if line[1] not in skipping_track_ids]
        for line in lines:
            (T, identifier, left, top, right, bottom, status) = line
            if T not in faces_by_time:
                faces_by_time[T] = []
            faces_by_time[T].append((identifier, (left, top, right, bottom)))
    return faces_by_time


def extract_tracks(video_file, tracks_file, labels, tracks_output_file, output_dir, visualize, model, cuda, feature_size):
    video = Video(video_file)
    frame_width, frame_height = video.frame_size

    face_by_time = read_tracks(tracks_file, labels)
    last_frame_faces = []
    track_frames = {}
    for timestamp, rgb in video:
        timestamp = "{:.3f}".format(timestamp)
        if timestamp not in face_by_time:
            continue
        faces = face_by_time[timestamp]
        for face in faces:
            tid, area = face
            if tid not in last_frame_faces:
                print(f'track {tid}: {timestamp} start extracting frames')

        last_frame_faces = []
        for face in faces:
            tid, area = face
            last_frame_faces.append(tid)
            try:
                image = get_face_crop(rgb, area, frame_height, frame_width)
                img_name = '{0}_{1}.jpg'.format(tid, timestamp)
                img_path = path.join(output_dir, "frames", img_name)
                cv2.imwrite(img_path, image)
                if tid not in track_frames:
                    track_frames[tid] = []
                track_frames[tid].append((timestamp, img_path))
                if visualize:
                    cv2.imshow('image_{}'.format(tid), image)
                    cv2.waitKey(1)
            except Exception as e:
                print('failed writing face_crop image', e, '(track {0}, time {1})'.format(tid, timestamp))

    with open('tracks_frames.data', 'wb') as frames_file:
        pickle.dump(track_frames, frames_file)
    save_track_features(track_frames, labels, tracks_output_file, model, cuda, feature_size)


def save_track_features(track_frames, labels, tracks_output_file, model, cuda, features_size):
    t = Tracks(features_size)
    features_extractor = Img2Vec(cuda=cuda, model=model, layer_output_size=features_size)
    for tid in track_frames:
        track_images = []
        timestamp, img_path = track_frames[tid][0]
        print(f'track {tid}: {timestamp} start extracting features')
        for track_frame in track_frames[tid]:
            timestamp, img_path = track_frame
            track_images.append(get_image(img_path))
        track_features = features_extractor.get_vec(track_images)
        for i in range(len(track_features)):
            f = track_features[i]
            timestamp, img_path = track_frames[tid][i]
            label = ""
            if tid in labels:
                label = labels[tid]
            t.add_track_features(tid, timestamp, f, label, img_path)

    t.save(tracks_output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', metavar='video_path', type=str, nargs='?',
                        default='data/TheBigBangTheory.mkv',
                        help='the path of the video')
    parser.add_argument('--tracks', metavar='tracks_path', type=str, nargs='?',
                        default='data/TheBigBangTheory.track.txt',
                        help='the path of the tracks')
    parser.add_argument('--output', metavar='output_path', type=str, nargs='?',
                        default='data',
                        help='the output directory')
    parser.add_argument('--tracks_output', metavar='tracks_output', type=str, nargs='?',
                        default='tracks.data',
                        help='the path of the tracks')
    parser.add_argument('--labels', metavar='labels', type=str, nargs='?',
                        default='',
                        help='the path of the labels file')
    parser.add_argument('--track_frames_path', metavar='track_frames_path', type=str, nargs='?',
                        help='the path of the labels file')
    parser.add_argument('--model', metavar='model', type=str, nargs='?', default='resnet50',
                        help='the model we want to use for feature extraction')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--feature_size', type=int, default=2048)

    args = parser.parse_args()

    video_file = args.video
    tracks_file = args.tracks
    output_dir = args.output
    labels_file = args.labels
    tracks_output_file = args.tracks_output

    print('track extraction input:')
    print('\tvideo:', video_file)
    print('\ttracks:', tracks_file)
    print('\tlabels:', labels_file)
    base_video_name = os.path.splitext(os.path.basename(video_file))[0]
    tracks_output_path = path.join(output_dir, '{}.{}'.format(base_video_name, tracks_output_file))
    print('track extraction writing output to:\n\t', tracks_output_path)
    labels = get_tracks_labels(labels_file)

    if args.track_frames_path is not None:
        with open(args.track_frames_path, 'rb') as track_frames_file:
            track_frames = pickle.load(track_frames_file)
            save_track_features(track_frames, labels, tracks_output_file, args.model, args.cuda, args.feature_size)
    else:
        extract_tracks(video_file, tracks_file, labels, tracks_output_path, output_dir, args.visualize, args.model, args.cuda, args.feature_size)


if __name__ == '__main__':
    main()
