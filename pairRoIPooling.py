class PairRoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(PairRoIPool, self).__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, roi_1s, roi_2s):
        # to pool feature map with given two rois
        # shape of roi_1 and roi_2: batch * N * 4
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = 1
        outputs = Variable(torch.zeros(batch_size,num_rois, num_channels, self.pooled_height, self.pooled_width))#.cuda()

        for batch_ind in range(batch_size):
            for roi_ind in range(num_rois):
                roi_1 = roi_1s[batch_ind][roi_ind]
                roi_2 = roi_2s[batch_ind][roi_ind]
                roi_1_start_w, roi_1_start_h, roi_1_end_w, roi_1_end_h = np.round(
                    roi_1[0:].data.cpu().numpy() * self.spatial_scale).astype(int)
                roi_2_start_w, roi_2_start_h, roi_2_end_w, roi_2_end_h = np.round(
                    roi_2[0:].data.cpu().numpy() * self.spatial_scale).astype(int)

                roi_start_w = min(roi_1_start_w, roi_2_start_w)
                roi_start_h = min(roi_1_start_h, roi_2_start_h)
                roi_end_w = max(roi_1_end_w, roi_2_end_w)
                roi_end_h = max(roi_1_end_h, roi_2_end_h)

                roi_width = max(roi_end_w - roi_start_w + 1, 1)
                roi_height = max(roi_end_h - roi_start_h + 1, 1)
                bin_size_w = float(roi_width) / float(self.pooled_width)
                bin_size_h = float(roi_height) / float(self.pooled_height)

                #new feature map with pixels that are not in ROI boxes set to zero
                feature = features[batch_ind]
                new_feature = torch.zeros(num_channels, data_height, data_width)
                new_feature[:,roi_1_start_w:roi_1_end_w, roi_1_start_h:roi_1_end_h]
                    = feature[:,roi_1_start_w:roi_1_end_w, roi_1_start_h:roi_1_end_h]
                new_feature[:,roi_2_start_w:roi_2_end_w, roi_2_start_h:roi_2_end_h]
                    = feature[:,roi_2_start_w:roi_2_end_w, roi_2_start_h:roi_2_end_h]
                

                for ph in range(self.pooled_height):
                    hstart = int(np.floor(ph * bin_size_h))
                    hend = int(np.ceil(ph + 1) * bin_size_h)
                    hstart = min(data_height, max(0, hstart + roi_start_h))
                    hend = min(data_height, max(0, hend + roi_start_h))
                    for pw in range(self.pooled_width):
                        wstart = int(np.floor(pw * bin_size_w))
                        wend = int(np.ceil((pw + 1) * bin_size_w))
                        wstart = min(data_width, max(0, wstart + roi_start_w))
                        wend = min(data_width, max(0, wend + roi_start_w))

                        is_empty = (hend <= hstart) or (wend <= wstart)
                        if is_empty:
                            outputs[batch_ind,roi_ind, :, ph, pw] = 0
                        else:
                            #data = features[batch_ind]
                            data = new_feature
                            outputs[batch_ind, roi_ind, :, ph, pw] = torch.max(
                                torch.max(
                                    data[:, hstart:hend, wstart:wend], 1
                                #)[0], 2)[0].view(-1)
                                )[0], 1)[0].view(-1)
        return outputs
