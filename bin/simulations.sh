#!/bin/bash

if [ ! -f bin/create_results.sh ]; then
  echo 'bin/create_results.sh is not within the current directory tree.'
  exit 1
fi

IFS=',' read -ra IN <<< "$1"

outFolder=output_$(date -d "today" +"%Y%m%d%H%M%S")
mkdir $outFolder

outFolders=("1_128x128_NoCarcin" "2_64x64_AllCarcin_Func2" "3_128x128_AllCarcin_Func2" "4_256x256_AllCarcin_Func2" \
            "5_512x512_AllCarcin_Func2" "6_256x256_AllCarcin_Func1" "7_256x256_AllCarcin_Func3" "8_256x256_AllCarcin_PDE" \
            "9_256x256_Carcin0_Func2" "10_256x256_Carcin1_Func2" "11_256x256_AllCarcin_Func2_Carcin0TimeInflux2Days-TimeNoInflux5Days" \
            "12_256x256_Carcin1_Func2_TimeInflux2Days-TimeNoInflux5Days" "13_256x256_Carcin1_Func2_TimeInflux5Days-TimeNoInflux2Days" \
            "14_256x256_AllCarcin_Func2_PerfectExcision_3Months_KeepField" "15_256x256_AllCarcin_Func2_PerfectExcision_6Months_KeepField" \
            "16_256x256_AllCarcin_Func2_PerfectExcision_9Months_KeepField" "17_256x256_AllCarcin_Func2_PerfectExcision_1Year_KeepField" \
            "18_256x256_AllCarcin_Func2_PerfectExcision_1Year3Months_KeepField" "19_256x256_AllCarcin_Func2_PerfectExcision_1Year6Months_KeepField" \
            "20_256x256_AllCarcin_Func2_PerfectExcision_1Year9Months_KeepField" "21_256x256_AllCarcin_Func2_PerfectExcision_2Years_KeepField" \
            "22_256x256_AllCarcin_Func2_PerfectExcision_2Years3Months_KeepField" "23_256x256_AllCarcin_Func2_PerfectExcision_2Years6Months_KeepField" \
            "24_256x256_AllCarcin_Func2_PerfectExcision_2Years9Months_KeepField" "25_256x256_AllCarcin_Func2_PerfectExcision_3Years_KeepField" \
            "26_256x256_AllCarcin_Func2_PerfectExcision_3Months_RemoveField" "27_256x256_AllCarcin_Func2_PerfectExcision_6Months_RemoveField" \
            "28_256x256_AllCarcin_Func2_PerfectExcision_9Months_RemoveField" "29_256x256_AllCarcin_Func2_PerfectExcision_1Year_RemoveField" \
            "30_256x256_AllCarcin_Func2_PerfectExcision_1Year3Months_RemoveField" "31_256x256_AllCarcin_Func2_PerfectExcision_1Year6Months_RemoveField" \
            "32_256x256_AllCarcin_Func2_PerfectExcision_1Year9Months_RemoveField" "33_256x256_AllCarcin_Func2_PerfectExcision_2Years_RemoveField" \
            "34_256x256_AllCarcin_Func2_PerfectExcision_2Years3Months_RemoveField" "35_256x256_AllCarcin_Func2_PerfectExcision_2Years6Months_RemoveField" \
            "36_256x256_AllCarcin_Func2_PerfectExcision_2Years9Months_RemoveField" "37_256x256_AllCarcin_Func2_PerfectExcision_3Years_RemoveField")

              # 1 Equilibrium
simulations=("bin/create_results.sh -s -t 8766 -g 128 -i 3 -n 1 -f $outFolder/${outFolders[0]}" \
              # 2 Grid Size 64x64
              "bin/create_results.sh -s -t 8766 -g 64 -i 2 -h 2 -h 2 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[1]}" \
              # 3 Grid Size 128x128
              "bin/create_results.sh -s -t 8766 -g 128 -i 2 -h 2 -h 2 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[2]}" \
              # 4 Grid Size 256x256, Carcinogen Function 2 test
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[3]}" \
              # 5 Grid Size 512x512
              "bin/create_results.sh -s -t 8766 -g 512 -i 2 -h 2 -h 2 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[4]}" \
              # 6 Carcinogen Function 1
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 1 -h 1 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[5]}" \
              # 7 Carcinogen Function 3
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 3 -h 3 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[6]}" \
              # 8 Carcinogen PDE
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 0 -h 0 -o 1 -o 1 -n 1 -f $outFolder/${outFolders[7]}" \
              # 9 Carcinogen 0 (Ethanol)
              "bin/create_results.sh -s -t 8766 -g 256 -i 0 -c 0 -h 2 -o 0 -n 1 -f $outFolder/${outFolders[8]}" \
              # 10 Carcinogen 1 (Nicotine)
              "bin/create_results.sh -s -t 8766 -g 256 -i 0 -c 1 -h 2 -h 2 -o 0 -o 0 -n 1 -f $outFolder/${outFolders[9]}" \
              # 11 Cyclic Carcinogen Onslaught: drinking 2 days a week and smoking everyday
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -a 2 -b 5 -n 1 -f $outFolder/${outFolders[10]}" \
              # 12 Cyclic Carcinogen Onslaught: smoking 2 days a week
              "bin/create_results.sh -s -t 8766 -g 256 -i 0 -c 1 -h 2 -h 2 -o 0 -o 0 -a -1 -b -1 -a 2 -b 5 -n 1 -f $outFolder/${outFolders[11]}" \
              # 13 Cyclic Carcinogen Onslaught: smoking 5 days a week
              "bin/create_results.sh -s -t 8766 -g 256 -i 0 -c 1 -h 2 -h 2 -o 0 -o 0 -a -1 -b -1 -a 5 -b 2 -n 1 -f $outFolder/${outFolders[12]}" \
              # 14 Excision at 3 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 220 -u 2 -p -n 1 -f $outFolder/${outFolders[13]}" \
              # 15 Excision at 6 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 439 -u 2 -p -n 1 -f $outFolder/${outFolders[14]}" \
              # 16 Excision at 9 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 658 -u 2 -p -n 1 -f $outFolder/${outFolders[15]}" \
              # 17 Excision at 1 year after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 877 -u 2 -p -n 1 -f $outFolder/${outFolders[16]}" \
              # 18 Excision at 1 year and 3 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1096 -u 2 -p -n 1 -f $outFolder/${outFolders[17]}" \
              # 19 Excision at 1 year and 6 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1315 -u 2 -p -n 1 -f $outFolder/${outFolders[18]}" \
              # 20 Excision at 1 year and 9 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1534 -u 2 -p -n 1 -f $outFolder/${outFolders[19]}" \
              # 21 Excision at 2 years after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1754 -u 2 -p -n 1 -f $outFolder/${outFolders[20]}" \
              # 22 Excision at 2 years and 3 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1974 -u 2 -p -n 1 -f $outFolder/${outFolders[21]}" \
              # 23 Excision at 2 years and 6 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 2194 -u 2 -p -n 1 -f $outFolder/${outFolders[22]}" \
              # 24 Excision at 2 years and 9 months after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 2414 -u 2 -p -n 1 -f $outFolder/${outFolders[23]}" \
              # 25 Excision at 3 years after first TC, Keep Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 2634 -u 2 -p -n 1 -f $outFolder/${outFolders[24]}" \
              # 26 Excision at 3 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 220 -u 2 -p -q -n 1 -f $outFolder/${outFolders[25]}" \
              # 27 Excision at 6 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 439 -u 2 -p -q -n 1 -f $outFolder/${outFolders[26]}" \
              # 28 Excision at 9 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 658 -u 2 -p -q -n 1 -f $outFolder/${outFolders[27]}" \
              # 29 Excision at 1 year after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 877 -u 2 -p -q -n 1 -f $outFolder/${outFolders[28]}" \
              # 30 Excision at 1 year and 3 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1096 -u 2 -p -q -n 1 -f $outFolder/${outFolders[29]}" \
              # 31 Excision at 1 year and 6 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1315 -u 2 -p -q -n 1 -f $outFolder/${outFolders[30]}" \
              # 32 Excision at 1 year and 9 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1534 -u 2 -p -q -n 1 -f $outFolder/${outFolders[31]}" \
              # 33 Excision at 2 years after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1754 -u 2 -p -q -n 1 -f $outFolder/${outFolders[32]}" \
              # 34 Excision at 2 years and 3 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 1974 -u 2 -p -q -n 1 -f $outFolder/${outFolders[33]}" \
              # 35 Excision at 2 years and 6 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 2194 -u 2 -p -q -n 1 -f $outFolder/${outFolders[34]}" \
              # 36 Excision at 2 years and 9 months after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 2414 -u 2 -p -q -n 1 -f $outFolder/${outFolders[35]}" \
              # 37 Excision at 3 years after first TC, Remove Field
              "bin/create_results.sh -s -t 8766 -g 256 -i 2 -h 2 -h 2 -o 0 -o 0 -j 2634 -u 2 -p -q -n 1 -f $outFolder/${outFolders[36]}")

for i in "${!simulations[@]}"; do
  if [[ " ${IN[@]} " =~ " $((i+1)) " ]] || [ $# -eq 0 ]; then
    echo "Running simulation $((i+1))"
    ${simulations[i]}
    if [ $? -eq 1 ]; then
      exit 1
    fi
    echo "Done simulation $((i+1))"
  fi
done

exit 0