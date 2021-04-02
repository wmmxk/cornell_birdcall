#!/bin/bash
# to create a template for checking one single thing in a function
# $1 is name.py
file=$1
file_path="../check_before_plugin/"$file
touch $file_path
echo -e "\"\"\"\n\n\"\"\"\n\n" > $file_path


echo "def fun():" >> $file_path
echo -e "    pass\n\n" >> $file_path
echo -e "if __name__ == \"__main__\":" >> $file_path
echo -e "    fun()\n" >> $file_path
