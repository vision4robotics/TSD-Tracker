function dpmr = dpmr(response,rate)
maxresponse = max(response(:));
range = ceil(sqrt(numel(response))*rate/2);

response = fftshift(response);

[xx, yy] = find(response == maxresponse, 1);
idx = xx-range:xx+range;
idy = yy-range:yy+range;
idy(idy<1)=1;idx(idx<1)=1;
idy(idy>size(response,2))=size(response,2);idx(idx>size(response,1))=size(response,1);

heigh_area = response(idx,idy);
response(idx,idy)=0;

molecule = maxresponse - min(heigh_area(:));
denominator = sum(response(:))/(numel(response)-numel(heigh_area)) - min(response(:));
dpmr = molecule/denominator;

end