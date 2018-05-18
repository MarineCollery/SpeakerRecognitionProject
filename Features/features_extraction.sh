#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`

if [ $# -ne 2 ] 
   then echo "${red}Enter the path of the .wav silence file ${reset}"
   		read path_wav
   		echo "${red}Enter the path of the desired output directory for the .csv files${reset}"
   		read path_csv
else
    path_wav=$1 
    path_csv=$2
fi

echo "${blue}File : ${path_wav}${reset}";

file_name=$(basename $path_wav)
file_name_witout_ext=${file_name%.*}


# /Users/Marine/Downloads/openSMILE-2.1.0/inst/bin/SMILExtract  -C /Users/Marine/Downloads/openSMILE-2.1.0/myconfig/MFCC12_E_D_A_Z_csv.conf  -I ${path_wav}  -O ${path_csv}/MFCC/${file_name_witout_ext}.csv
/Users/Marine/Downloads/openSMILE-2.1.0/inst/bin/SMILExtract  -C /Users/Marine/Downloads/openSMILE-2.1.0/myconfig/Speaker_recognition.conf  -I ${path_wav}  -O ${path_csv}/Speaker_trait/${file_name_witout_ext}.csv