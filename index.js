import * as faceapi from 'face-api.js';

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons';

async function run() {

    console.clear()

  await faceDetectionNet.loadFromDisk('./weights')
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights')
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./weights')

  const twoFaces = await canvas.loadImage('./imgTeste.jpg');

  const detectionss = await faceapi.detectAllFaces(twoFaces, faceDetectionOptions);
  const faceImages = await faceapi.extractFaces(twoFaces, detectionss)

  let count = 0;
  try{
    faceImages.map(img => {
        count++;
        saveFile(`faceDetection${count}.jpg`, img.toBuffer('image/jpeg'))
      });
  }catch(e){
      console.log(e)
  }
  
  const img1 = await canvas.loadImage('./out/faceDetection1.jpg')
  const img2 = await canvas.loadImage('./out/faceDetection2.jpg')

  const descriptor1 = await faceapi.computeFaceDescriptor(img1, faceDetectionOptions);
  const descriptor2 = await faceapi.computeFaceDescriptor(img2, faceDetectionOptions);

  const distance = faceapi.euclideanDistance(descriptor1, descriptor2).toFixed(3)

  console.log(distance)
 
}

run()