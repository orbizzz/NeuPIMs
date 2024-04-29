#!/usr/bin/env fish

set fnames (ls)
set dnames 

for fname in $fnames
	if string match -q "*.csv" $fname
		set dname (string sub --end=-6 $fname)
		if not contains $dname $dnames
			set -a dnames $dname
			mkdir -p $dname
		end
		mv $fname $dname
	end
end

echo $dnames
