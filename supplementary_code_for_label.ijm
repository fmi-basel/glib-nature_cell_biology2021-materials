name = getTitle; 
dotIndex = indexOf(name, "."); 
title = substring(name, 0, dotIndex); 

append = "_label.tif"

title = title + append

newImage(title, "16-bit black", getWidth(), getHeight(), 1);

for (index = 0; index < roiManager("count"); index++) {
	roiManager("select", index);
	setColor(index+1);
	fill();
}

resetMinAndMax();
run("glasbey");
