# img2d6-cuda

This project  implements an program written in CUDA that transforms an input image into a grid of D6 dices, 
where each die face represents a block of the original image. The program uses the CUDA parallel computing framework to process the image efficiently.

To compile: 
  - nvcc -o image2d6 main.cu

There are two options: 
 - SIMPLE: ./image2d6 image.jpg SIMPLE
 
  ![image](https://github.com/user-attachments/assets/9d7fa6fd-688e-47ec-898c-26ee7af9c15b) 
  ![image](https://github.com/user-attachments/assets/1cd050c9-7574-44a1-985c-3b5d9ae4f1ab) 
  ![image](https://github.com/user-attachments/assets/13adb883-158a-459c-baf2-2d855428ca4b) 

  - DETAIL ./image2d6 image.jpg DETAIL

  ![image](https://github.com/user-attachments/assets/5875d04d-bd16-49ad-8fe0-5ee7c016367c)
  ![image](https://github.com/user-attachments/assets/9673f62c-e558-47f2-8e90-5581c3246de5)
  ![image](https://github.com/user-attachments/assets/41e07dc5-7d9c-4c1c-b190-5c586299db4d)
