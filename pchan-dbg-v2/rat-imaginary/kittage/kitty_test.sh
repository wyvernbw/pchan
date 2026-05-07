#!/usr/bin/bash

# get the file location from which we can read the stuff that the cargo test parent process wrote in
fifo="${KITTYIMG_PIPE}"

input=""
while [ "${#input}" -eq 0 ]
do
	# and then assign it to the variable
	input="$(cat "$fifo")"
done

# output it to kitty, which this should be running inside
echo -en "$input"

# and then take the stdin and pass it to the fd handle, which the cargo test process will read back from
stty raw -echo min 0 time 5
read -r response

echo "$response" | cut -d "$(echo -en '\e')" -f2 > "$fifo"
rm "$fifo"
