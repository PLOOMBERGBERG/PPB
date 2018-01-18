function [move_r,move_c,normcor] = real_coherent(image1,image2,figshow)
% 逐点的滑动窗相关计算转化为图像块之间的互相关
figshow = 0;
%% 实相干系数计算（FFT）
[nr,nc] = size(image1); %假设image1与image2大小相同

im1abs = abs(image1);
im2abs = abs(image2);

im1fft = fft2(im1abs);
im2fft = fft2(im2abs);
cor = abs(fftshift(ifft2(im1fft.*conj(im2fft))));
normcor = cor./max(max(cor));
[r,c] = deal(fix(nr/2)+1,fix(nc/2)+1);
[r1,c1,~] = find(normcor == 1);
move_r = r - r1;
move_c = c - c1;
%% 输出
if figshow == 1
    figure;
    imagesc(normcor)
    colormap(jet)
    colorbar
    title('real coherence coefficient','fontWeight','Bold')
end
end