def im_pre_process(img, params):
#%IM_PRE_PROCESS pre-process the image.

#% Copyright 2015-2016 Nadith Pathirage, Curtin University.

#    if (~isfield(params, 'histeq')); params.histeq = false; end
#    if (~isfield(params, 'normalize')); params.normalize = false; end
#    if (~isfield(params, 'histmatch')); params.histmatch = false; end
#    if (~isfield(params, 'cann_img')); params.cann_img = []; end

#    if (params.histeq)
#        if (size(img, 3) == 3)
#            img = histeqRGB(img);
#        else
#            img = histeq(img);
#        end
#    end

#    if (params.normalize)        
#        img = bsxfun(@minus, double(img), mean(mean(img))); 
#        img = bsxfun(@rdivide, double(img), std(std(img))); 
#    %         means = mean(mean(low_dim_img));
#    %         for i=1:numel(means)
#    %             img(:, :, i) = img(:, :, i) - means(i);
#    %         end
#    end
    
#    % Histrogram matching
#    if (params.histmatch && ~isempty(params.cann_img))            
#        img = histmatch(params.cann_img, img);
#    end
        
    return img

