function [unwrapped_phase] = QualityGuidedUnwrap2D(wrapped_phase,im_phase_quality,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QualityGuidedUnwrap2D implements 2D quality guided path following phase
% unwrapping algorithm.
% Inputs:  1. Complex image in .mat double format
%          2. Binary mask (optional)          
% Outputs: 1. Unwrapped phase image
%          2. Phase quality map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n]    = size(wrapped_phase);
im_mag   = ones(m,n);           %Magnitude image
im_phase = wrapped_phase;       %Phase image
% -----------------------------------------------------------
%% Replace with your mask (if required)
mag_max = max(im_mag(:));
indx1 = find(im_mag < 0.1*mag_max);  %Intensity = mag^2, so this = .04 threshold on the intensity
im_mask = ones(size(wrapped_phase));
im_mask(indx1) = 0;                  %Mask
if(~exist('im_mask','var'))
  im_mask = ones(size(wrapped_phase));          %Mask (if applicable)
end
% figure; imagesc(im_mag.*im_mask),   colormap(gray), axis square, axis off, title('Initial masked magnitude'); colorbar;
% figure; imagesc(im_phase.*im_mask), colormap(gray), axis square, axis off, title('Initial masked phase'); colorbar;

unwrapped_phase = zeros(size(wrapped_phase));        %Initialze the output unwrapped version of the phase
adjoin = zeros(size(wrapped_phase));            %Zero starting matrix for adjoin matrix
unwrapped_binary = zeros(size(wrapped_phase));  %Binary image to mark unwrapped pixels
%% �в�����
display = 0;
[~,residuemat] = Calculation_Residues(im_phase,display);
%% Automatically (default) or manually identify starting 
%% seed point on a phase quality map 
minp = im_phase_quality(2:end-1, 2:end-1); minp = min(minp(:));
% maxp = im_phase_quality(2:end-1, 2:end-1); maxp = max(maxp(:));

if(1)  %1��ȡ��λƫ����С�ĵ����ڵ����½ǵĵ㣬0��ȡ���Ͻǵ�һ���㡣
  [rowrefn,colrefn] = find(im_phase_quality(2:end-1, 2:end-1) == minp);
  colref = colrefn(1)+1;
  rowref = rowrefn(1)+1;
  if colref>n
      colref = n-1;
  elseif rowref>m
      rowref = m-1;
  end 
else   % Chose starting point = max. intensity, but avoid an edge pixel
  [rowrefn,colrefn] = find(im_mag(2:end-1, 2:end-1) >= 0.99*mag_max);%��im_magΪȫ1����ʱ���õ�Ϊ���ϵ�һ��
  % Ϊ�˱���ȡ����Ե�㣬�������кŸ���1
  rowref = rowrefn(1)+1; % choose the 1st point for a reference (known good value)
  colref = colrefn(1)+1; % choose the 1st point for a reference (known good value)
end
%% Unwrap
unwrapped_phase(rowref,colref) = im_phase(rowref,colref);  %                        %Save the unwrapped values
unwrapped_binary(rowref,colref,1) = 1; %�������Ǿ�����չΪ��ά���ҽ�seed�㴦���Ϊ1�������ѽ��
% �����ھ�����չΪ��ά���󣬽�seed������ڵ������ھ����б��Ϊ1�����ھ����ʼΪ�㣩
if im_mask(rowref-1, colref, 1)==1;  adjoin(rowref-1, colref, 1) = 1; end %Mark the pixels adjoining the selected point��seed point��
if im_mask(rowref+1, colref, 1)==1;  adjoin(rowref+1, colref, 1) = 1; end
if im_mask(rowref, colref-1, 1)==1;  adjoin(rowref, colref-1, 1) = 1; end
if im_mask(rowref, colref+1, 1)==1;  adjoin(rowref, colref+1, 1) = 1; end
% [unwrapped_phase] = guided_flood_fill(im_phase, im_mag, unwrapped_phase, unwrapped_binary, im_phase_quality, adjoin, im_mask,residuemat);    %Unwrap
unwrapped_phase = GuidedFloodFill_r1(im_phase, im_mag, unwrapped_phase, unwrapped_binary, im_phase_quality, adjoin, im_mask,residuemat,C);
end
