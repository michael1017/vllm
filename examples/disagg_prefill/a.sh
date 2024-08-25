#!/bin/bash  

repeat_string() {  

    local str="$1"

    local times="$2"
    local result=""

    for (( i=0; i<$times; i++ )); do  

        result+="$str"  
    done  
    echo "$result"
}  

str="abc"  
times=3  
repeated_str=$(repeat_string "$str" "$times")  
echo "$repeated_str"