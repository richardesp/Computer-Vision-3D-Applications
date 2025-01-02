# Como realizar la calibración estereo

En primer lugar, deberá seguirse el protocolo estándar de C++ de cmakelist: 

``sh
mkdir build; cd build; cmake ..; make
``

``sh
./stereo_checkundistorted ../stereo/calibration/m001.jpg ../stereo/stereoparms.yml
``

