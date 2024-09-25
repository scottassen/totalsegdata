function GeoFilter3D_DICOM()
clc ; close all
% Get directory listing
prompt = 'Enter DICOM directory name? ';
dicom_directory=input(prompt,'s'); 
% listing = dir(dicom_directory);
% % Number of files
% N = numel(listing);
% fprintf('Number of Files in %s is %d\n',dicom_directory,N);
sourcetable = dicomCollection(dicom_directory);
disp(sourcetable);
prompt = 'Enter enter source? ';
source=input(prompt,'s'); 
V = dicomreadVolume(sourcetable,source,'MakeIsotropic',false);
V = squeeze(V);
fprintf('----->Source %s %d %d %d\n',source,size(V));
%% Display input volume
volumeViewer(V);
%% Enter filter parameters
prompt = 'Enter neigbourhood support WL [3-21]? ';
WL=input(prompt);
prompt = 'Enter neigbourhood support WC [3-21]? ';
WC=input(prompt);
prompt = 'Enter neigbourhood support WS [3-21]? ';
WS=input(prompt);
alpha=1.0;
prompt = 'Enter sigma value? ';
sigma = input(prompt);
%% Filter Volume
V_fil=Geodesic3DFilter_Vol(V,WL,WC,WS,sigma,alpha);
%% Save filtered volume 
prompt = 'Enter output DICOM directory name? ';
o_dir=input(prompt,'s'); 
dicomwritevolume(o_dir,V_fil);
%% Dipslay filtered version
volumeViewer(V_fil);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function V_fil = Geodesic3DFilter_Vol(V,WL,WC,WS,sigma,alpha)

% Design the Kernel

sz=size(V);
lines=sz(1,1);
columns=sz(1,2);
depth=sz(1,3);
fprintf(' line %d Columns %d channel %d\n',lines,columns,depth);
scalex=1.0;
scaley=1.0;
scalez=1.0;

M = (WL - 1)/ 2;
N = (WC - 1)/ 2;
S = (WS - 1)/2;

fV_fil=zeros(lines,columns,depth,'single');
V_fil=V;
wind=zeros(WL,WC,WS,1,'single');
weights=zeros(WL,WC,WS,'single');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start the Convolution
    fprintf('Start Convolution... \n')
  for s=1+S:depth-S
   fprintf('Layer %d\n',s);
    for i = 1+M : lines-M
        for j = 1+N : columns-N

%            fprintf(' i= %d j= %d k= %d\n',i,j,s);
                for l = -M : M
                    for m = -N : N
                      for k=-S : S
                        wind(l+M+1, m+N+1, k+S+1,1) = (i+l)*scalex;
                        wind(l+M+1, m+N+1, k+S+1,2) = (j+m)*scaley;
                        wind(l+M+1, m+N+1, k+S+1,3) = (k+s)*scalez;
                        wind(l+M+1, m+N+1, k+S+1,4) = cast(V(i+l,j+m,s+k),'single');
                      end
                    end
                 end

                % Compute Manifold Distances
                [dis,~] = Volume_Dist3D_Vol(wind, WL, WC,WS, alpha);

                sum_weight = 0;

                for k = 1 : WS
%                     fprintf('k= %d\n',k);
                    for l = 1 : WL
                     for m=1 : WC
                        Dis = dis(l,m,k).^2;
                        Sigma = 2*sigma*sigma;
                        weights(l,m,k) = exp(-Dis/Sigma);
                        sum_weight = sum_weight + weights(l,m,k);
%                          fprintf('%2.1f ',dis(l,m,k));
                     end
%                     fprintf('\n');
                    end

                end
           
% Perfom Weighted Average

                fV_fil(i,j,s)= 0.0; 
                for l = -M : M
                    for m = -N : N 
                     for k = -S : S
                        fV_fil(i,j,s) = fV_fil(i,j,s) + weights(l+M+1,m+N+1,k+S+1)*wind(l+M+1,m+N+1,k+S+1,4);
                     end
                    end
                end
                fV_fil(i,j,s)=fV_fil(i,j,s)/sum_weight;             
            end
     end
end
for k=1+S:depth-S
    for i = 1+M : lines-M
       for j = 1+N : columns-N
           V_fil(i,j,k)=fV_fil(i,j,k);  
        end
    end
end
return

end
% Compute Intrinsic Distances within the given volume window by 
% Using the Single-Source Shortest Path Algorithm
% INPUT
% Wind(WL,WC,WS,1) : X,Y,Z,val components of the pixels in the window
% WL, WC, WS : Size of the filtering volume Window
% OUTPUT
% dis(WL,WC,WS) : Intrinsic distances from the center pixel
% Prec(WL,WC,WS,2) : Backward trajectory of geodesic

function [dis, Prec] = Volume_Dist3D_Vol(Wind,  WL, WC, WS, alpha)
Vmax = 100.00;
MI = (WL-1)/2;
MJ = (WC-1)/2;
MK = (WS-1)/2;

LOC=zeros(WC*WL*WS,'uint32');
ELM=zeros(WC*WL*WS,'uint32');
MARK=zeros(WC*WL*WS,'uint32');
SP=zeros(WC*WL*WS,'single');
dct=zeros(WC*WL*WS,3,'int32');
dis=zeros(WC*WL*WS,'single');
% Initialization of the Heap, SP
for k=-MK:MK
   for i = -MI:MI
    for j = -MJ:MJ

        icode = iencode3D(i, j,k, MI, MJ,MK);
        dct(icode,1)=i;
        dct(icode,2)=j;
        dct(icode,3)=k;
        [iw,jw,kw] = idecode3D(icode,dct);
%        fprintf('icode %d i %d j %d k %d\n',icode,iw,jw,kw);
        SP(icode) = Vmax;
        LOC(icode) = icode;
        ELM(icode) = icode;
        MARK(icode) = 0;
     end
    end
end
%fprintf('Intitialize Heap\n');
SP(1) = 0.0;
icode = iencode3D(0, 0,0, MI, MJ,MK);
elm1 = ELM(1);
iloc = LOC(icode);
LOC(icode) = 1;
ELM(1) = icode;
ELM(iloc) = elm1;
LOC(elm1) = iloc;

Prec(MI+1, MJ+1, MK+1, 1) = 0;
Prec(MI+1, MJ+1, MK+1, 2) = 0;
Prec(MI+1, MJ+1, MK+1, 3) = 0;
%printheap(SP,nrest,LOC,ELM);
%fprintf('Start shorted part find\n');
for i = 1:WL*WC*WS-1
    nrest = WL*WC*WS-i+1;
    % Find the next shortest path
    icodew = ELM(1);
    spw = SP(1);
    [iw,jw,kw] = idecode3D(icodew,dct);
%   fprintf('icodew %d iw %d jw %d kw %d\n',icodew,iw,jw,kw);
%    printheap(SP,nrest,LOC,ELM);
    % Remove the first element from the heap SP
    MARK(icodew) = 1;
    sp1 = SP(1);
    elm1 = ELM(1);
    SP(1) = SP(nrest);
    ELM(1) = ELM(nrest);
    LOC(ELM(1)) = 1;
    SP(nrest) = sp1;
    ELM(nrest) = elm1;
    LOC(elm1) = nrest;
    % CALL SHIFT DOWN
%     fprintf('Before Shift Down\n');
%      printheap(SP,nrest,LOC,ELM);
    [SP,LOC,ELM]=shiftdown(SP,1,nrest-1,LOC,ELM);
%     fprintf ('After shift down\n');
%     printheap(SP,nrest,LOC,ELM);
    % First Element is Removed
    % Update the shortest distances in the heap, SP
   for kk=-1:1 
    for ii = -1:1
        for jj = -1:1
            iz = iw + ii;
            jz = jw + jj;
            kz = kw + kk;
            if (((iz >= -MI) && (iz <= MI)) && ((jz >= -MJ) && (jz <= MJ))&&((kz >= -MK) && (kz <= MK)))
                icodez = iencode3D(iz, jz, kz, MI, MJ, MK);
              if (MARK(icodez) == 0)
                spz = SP(LOC(icodez));
                dist = edge_length(iw+MI+1, jw+MJ+1,kw+MK+1, iz+MI+1,jz+MJ+1,kz+MK+1, Wind,alpha);
                if((spw + dist)< spz)
                % The distance is less than the current distance,
                % So update the heap
                SP(LOC(icodez)) = spw + dist;
%                fprintf('Dist %f\n',dist+spw);
          
    % CALLING SHIFTUP FUNCTION

                [SP,LOC,ELM]=shiftup(SP,LOC(icodez),LOC,ELM);
 %               printheap(SP,nrest,LOC,ELM);
                Prec(iz+MI+1, jz+MJ+1, kz+MK+1, 1) = iw + MI + 1;
                Prec(iz+MI+1, jz+MJ+1, kz+MK+1, 2) = iw + MI + 1;
                Prec(iz+MI+1, jz+MJ+1, kz+MK+1, 3) = kw + MK + 1;
                
                end
              end
            end
        end
    end
   end
end
    % Output the current Distance
%    printheap(SP,121,LOC,ELM);
for k = -MK:MK
  for j = -MJ:MJ
    for i = -MI:MI
        icode = iencode3D(i, j, k, MI, MJ, MK);
        dis(i+MI+1, j+MJ+1,k+MK+1) = SP(LOC(icode));
    end
  end
end
return
end

%% Compute the distance between 2 points (iw,jw,kw) and (iz,jz,kz)

function distance = edge_length(iw, jw,kw, iz, jz,kz, Wind, alpha)
d = 0.0;
for i = 1:3
    d = d + (Wind(iw, jw, kw, i) - Wind(iz, jz, kz, i)).^2;
end
    d = d + ((Wind(iw, jw, kw, 4) - Wind(iz, jz, kz, 4))/alpha).^2;
distance = sqrt(d);
end
%% Compute linear index of the window
%  INPUT
%  I,J  : index of the window -MI <= I <= MI, -MJ <= J <= MJ
%  MI,MJ
function icode = iencode3D(I,J,K,MI,MJ,MS)
icode = ((2*MJ+1)*(2*MI+1)*(K+MS))+(2*MJ+1)*(I+MI)+J+MJ+1;
return
end

%% decode the linear code into index of window
% INPUT
% icode : linear code
% MI,MJ
% OUTPUT
% I,J : index of window
function [i,j,k] = idecode3D(icode,dct)
i=dct(icode,1);
j=dct(icode,2);
k=dct(icode,3);
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,LOC,ELM]=shiftdown(H,is,ie,LOC,ELM)
% Shiftdown procedure for heap update
% Heap: H(1) is min
%
i = is;
j = 2*i;
x = H(i);
p = ELM(i);
while (j <= ie)
    if (j < ie)
        if (H(j) > H(j+1))
            j = j+1;
        end
    end
    if (x <= H(j))
        break;
    else
        H(i)=H(j);
        pp=ELM(j);
        ELM(i)=pp;
        LOC(pp)=i;
        i=j;
        j=2*i;
%        fprintf('i %d j %d\n',i,j);
    end
end
    H(i)=x;
    LOC(p)=i;
    ELM(i)=p;
return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
function [H,LOC,ELM]=shiftup(H,is,LOC,ELM)
    % Shiftup procedure for heap update
    % Heap: H(1) is min
    %
    
i=is;
j=floor(i/2);
x=H(i);
p=ELM(i);
while (j >= 1)
    if (H(j)<= x)
        break;
    end
        H(i)=H(j);
        pp=ELM(j);
        ELM(i)=pp;
        LOC(pp)=i;
        i=j;
        j=i/2;
        if(i==1)j=0;
        end
%        fprintf('**i %d j %d\n',i,j);
end
    H(i)=x;
    ELM(i)=p;
    LOC(p)=i;
return
end
function printheap(H,ie,LOC,ELM)
for i=1:ie
    fprintf('i = %d H= %f LOC = %d ELM= %d\n',i,H(i),LOC(ELM(i)),ELM(i));
end
return
end