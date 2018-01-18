function [move_r,move_c,normcor] = real_coherent(image1,image2,figshow)
% ���Ļ�������ؼ���ת��Ϊͼ���֮��Ļ����
figshow = 0;
%% ʵ���ϵ�����㣨FFT��
[nr,nc] = size(image1); %����image1��image2��С��ͬ

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
%% ���
if figshow == 1
    figure;
    imagesc(normcor)
    colormap(jet)
    colorbar
    title('real coherence coefficient','fontWeight','Bold')
end
end