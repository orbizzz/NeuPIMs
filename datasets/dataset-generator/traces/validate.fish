#!/usr/bin/env fish

for dname in *
	if test -d $dname
		set -l traces (ls $dname)
		string match -e $dname $traces
	end
end

