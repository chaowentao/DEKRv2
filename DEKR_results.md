
### Results on CrowdPose test without multi-scale test with rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18**           | 0.660 | 0.861 | 0.712 | 0.738 | 0.922 | 0.785 | 0.735 | 0.669 | 0.571 |
| **pose_hrnet_w32(paper)**    | 0.657 | 0.857 | 0.704 | 0.723 | 0.906 | 0.769 | 0.730 | 0.664 | 0.574 |
| **pose_hrnet_w18_dc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | 0.629 | 0.844 | 0.681 | 0.716 | 0.917 | 0.766 | 0.701 | 0.639 | 0.539 |
| **pose_hrnet_w18_gc2**       | 0.633 | 0.851 | 0.685 | 0.711 | 0.914 | 0.761 | 0.708 | 0.641 | 0.543 |
| **pose_hrnet_w18_gc3**       | 0.659 | 0.863 | 0.709 | 0.736 | 0.923 | 0.781 | 0.732 | 0.667 | 0.570 |
| **pose_hrnet_w18_part**      | 0.654 | 0.856 | 0.705 | 0.735 | 0.921 | 0.782 | 0.728 | 0.662 | 0.562 |
| **pose_hrnet_w18_part2**     | **0.666** | 0.863 | 0.715 | 0.742 | 0.924 | 0.789 | 0.739 | 0.673 | 0.579 |
| **pose_hrnet_w18_part3**     | 0.659 | 0.862 | 0.710 | 0.737 | 0.922 | 0.783 | 0.735 | 0.667 | 0.571 |

### Results on CrowdPose test without multi-scale test no rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)** | 0.648 | 0.850 | 0.701 | 0.738 | 0.922 | 0.784 | 0.720 | 0.658 | 0.558 |
| **pose_hrnet_w18_dc**        | 0.649 | 0.852 | 0.700 | 0.738 | 0.921 | 0.784 | 0.721 | 0.658 | 0.559 |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | 0.617 | 0.834 | 0.669 | 0.715 | 0.917 | 0.765 | 0.687 | 0.627 | 0.525 |
| **pose_hrnet_w18_gc2**       | 0.619 | 0.840 | 0.672 | 0.711 | 0.914 | 0.761 | 0.694 | 0.628 | 0.527 |
| **pose_hrnet_w18_gc3**       | 0.647 | 0.853 | 0.697 | 0.736 | 0.923 | 0.781 | 0.718 | 0.656 | 0.556 |
| **pose_hrnet_w18_part**      | 0.640 | 0.845 | 0.692 | 0.735 | 0.920 | 0.782 | 0.713 | 0.649 | 0.546 |
| **pose_hrnet_w18_part2**     | 0.654 | 0.854 | 0.704 | 0.742 | 0.923 | 0.789 | 0.727 | 0.662 | 0.566 |
| **pose_hrnet_w18_part3**     | 0.647 | 0.853 | 0.699 | 0.737 | 0.922 | 0.783 | 0.720 | 0.656 | 0.559 |

### Results on CrowdPose test with multi-scale test no rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)** | 0.659 | 0.830 | 0.714 | 0.753 | 0.919 | 0.803 | 0.745 | 0.670 | 0.549 |
| **pose_hrnet_w18_dc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc2**       | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc3**       | 0.659 | 0.834 | 0.711 | 0.751 | 0.920 | 0.799 | 0.748 | 0.672 | 0.545 |
| **pose_hrnet_w18_part**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part2**     | 0.661 | 0.833 | 0.716 | 0.754 | 0.919 | 0.803 | 0.748 | 0.672 | 0.552 |
| **pose_hrnet_w18_part3**     | - | - | - | - | - | - | - | - |

### Results on CrowdPose test with multi-scale test with rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)** | 0.674 | 0.851 | 0.729 | 0.754 | 0.921 | 0.804 | 0.761 | 0.685 | 0.566 |
| **pose_hrnet_w18_dc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc2**       | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc3**       | 0.674 | 0.852 | 0.727 | 0.753 | 0.922 | 0.801 | 0.762 | 0.686 | 0.561 |
| **pose_hrnet_w18_part**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part2**     | 0.676 | 0.852 | 0.730 | 0.756 | 0.921 | 0.805 | 0.762 | 0.686 | 0.570 |

### Results on CrowdPose test without multi-scale test no rescorenet 100 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)**  | 0.583 | 0.801 | 0.630 | 0.710 | 0.915 | 0.756 | 0.653 | 0.596 | 0.483 |
| **pose_hrnet_w18_dc**         | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**       | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**         | 0.558 | 0.800 | 0.604 | 0.679 | 0.906 | 0.728 | 0.625 | 0.570 | 0.464 |
| **pose_hrnet_w18_gc2**        | 0.562 | 0.809 | 0.606 | 0.673 | 0.904 | 0.719 | 0.636 | 0.572 | 0.469 |
| **pose_hrnet_w18_gc3**        | 0.602 | 0.822 | 0.647 | 0.703 | 0.910 | 0.747 | 0.678 | 0.613 | 0.503 |
| **pose_hrnet_w18_part**       | 0.585 | 0.808 | 0.632 | 0.703 | 0.913 | 0.750 | 0.657 | 0.597 | 0.484 |
| **pose_hrnet_w18_part2**      | 0.586 | 0.805 | 0.633 | 0.710 | 0.914 | 0.756 | 0.656 | 0.599 | 0.490 |
| **pose_hrnet_w18_part_final** | 0.588 | 0.817 | 0.633 | 0.703 | 0.912 | 0.748 | 0.664 | 0.598 | 0.492 |

### Results on CrowdPose test without multi-scale test with rescorenet 100 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)**  | 0.602 | 0.818 | 0.648 | 0.709 | 0.915 | 0.755 | 0.674 | 0.614 | 0.503 |
| **pose_hrnet_w18_dc**         | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**       | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**         | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc2**        | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc3**        | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part**       | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part2**      | 0.606 | 0.823 | 0.652 | 0.710 | 0.914 | 0.756 | 0.679 | 0.617 | 0.510 |
| **pose_hrnet_w18_part_final** | - | - | - | - | - | - | - | - | - |

实验参数说明：

1. **pose_hrnet_w18_dc**: 把adaptive convolution替换成自己改写的可变性形卷积
2. **pose_hrnet_w18_dc_3**: 替换backbone第三层的卷积类型为自己改写的可变性形卷积
3. **pose_hrnet_w18_gc**: 使用group convolution分组的方式替换现有分支结构
4. **pose_hrnet_w18_gc2**: 使用group convolution分组的方式替换现有分支结构，其中的adaptive conv也使用分组卷积
<!-- ![dekr_gc](https://github.com/chaowentao/images/blob/main/dekr_gc.png?raw=true) -->
5. **pose_hrnet_w18_part**: 按照给定规则对现有分组方式进行优化，比如分五组，adaptive conv没有使用分组卷积
<!-- ![dekr_part](https://github.com/chaowentao/images/blob/main/dekr_group.png?raw=true) -->




baseline

flops: 18.806G parms: 9.655M
total time: 174.50958895683289、
per image time: 0.17450958895683288

gc3
flops: 18.806G parms: 9.655M
total time: 173.67533683776855
per image time: 0.17367533683776856

gc2
flops: 18.803G parms: 9.654M
total time: 146.3964364528656
per image time: 0.1463964364528656

gc
flops: 18.803G parms: 9.654M
total time: 169.95253992080688
per image time: 0.1699525399208069