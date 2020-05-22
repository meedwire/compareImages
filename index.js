import * as faceapi from 'face-api.js';

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons';

async function run() {

    console.clear()

    await faceDetectionNet.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
    await faceapi.nets.faceRecognitionNet.loadFromDisk('./weights');

    const faces = await canvas.loadImage('./imgTeste.jpg');

    const results = await faceapi.detectAllFaces(faces, faceDetectionOptions)
                                    .withFaceLandmarks()
                                    .withFaceDescriptors();

    const descriptors = results.map(({descriptor}) => descriptor);

    const distance = faceapi.euclideanDistance(descriptors[0], descriptors[1]).toFixed(3);

    console.log(distance);

};

run();