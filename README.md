# Player-Localization-and-Field-Estimation
Player-Localization-and-Field-Estimation

## Requirments

- OpenCV >= 4.5.1  
- Numpy >= 1.19

## Setup
- Download the contents of this [Folder](https://drive.google.com/drive/folders/11KnQ3_e4tMpHY8YDdH5GY0REZhYtqnL6?usp=sharing) and place them inside SRC/Data   
- Download the contents of this [Folder](https://drive.google.com/drive/folders/15UXxo1Gp6eQcoIzsx1m64aP6EHvHs1wp?usp=sharing) and place them inside SRC/Model 


| Argument | Usage |
| ------ | ------ |
| Verbose | Development Mode, to print out intermediate data when needed. |
| GPU | When used, OpenCV will default to GPU. Otherwise CPU. |
| write | When used, the script will write the video output. |


To run the project:

```sh
python3 main.py --GPU --write
```


