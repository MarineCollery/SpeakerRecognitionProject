#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`

if [ $# -ne 2 ] 
   then echo "${red}Enter the path of the directory with the .wav files ${reset}"
   		read path_wav
   		echo "${red}Enter the path of the desired output directory for the .csv files${reset}"
   		read path_csv
else
    path_wav=$1 
    echo "${blue}Directory : ${path_wav}${reset}";
    path_csv=$2
    echo "${blue}Features saved in : ${path_csv}${reset}";
fi

echo "${blue}Directory : ${path_wav}${reset}";
echo "${blue}Features saved in : ${path_csv}${reset}";

wav_files=`find ${path_wav} -maxdepth 1 -name "*.wav"`
echo $wav_files;

mkdir ${path_csv}/MFCC
mkdir ${path_csv}/Speaker_trait

for line in $wav_files ; do
	bash features_extraction.sh ${line} ${path_csv}
done