************************************************************************************************
************************************************************************************************
******************************************CUDA POISSON SAMPLING LIBRARY:************************
************************************************************************************************
************************************************************************************************

This library was created under Bohemia Interactive and uses adjusted parallel algorithm based on paper Parallel Poisson Disk Sampling made by Li-Yi Wei.

Plugin requirements*:
 Windows OS
 Dedicated NVIDIA GPU
 

 *Since library was not properly tested on many different computers the requirements are just orientational.


Library was created on  CUDA V10.1.243. Full functionality on different versions is not guaranteed!

Library was created to make samples used for grass distribution in thesis of Matouš Procházka who is also author of this library. 

The library uses an external library not created nor owned by me called lodePNG made by Lode Vandevenne.


File framework.cu contains main testing environment which allows adjustments in order to get the desired results without necessity of intrusion within library code itself
	


 