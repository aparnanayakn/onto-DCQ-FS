#!/bin/sh
#
while IFS=, read -r one two three four; do 
    echo "The 1st field is $one, the 2nd $two and the 3rd $three"; 
    java -jar ./target/r2rml.jar ./$one ./$two ./$three $four
  done < testcases.csv
