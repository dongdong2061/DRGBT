from DRGBT603_dataset import DRGBT603

DRGBT603 = DRGBT603()

"""
LasHeR have 3 benchmarks: PR, NPR, SR
"""

# Register your tracker
# lasher(
#     tracker_name="tracker1",
#     result_path="/data1/Code/luandong/WWY_code_data/Codes/imgfuse_fusion2track/ostrack_fusion2track/temp", 
#     bbox_type="ltwh")
DRGBT603(
    tracker_name="tracker2",
    result_path="RGBT_workspace/results/DMET/DMET15", 
    bbox_type="ltwh")

# Evaluate multiple trackers
pr_dict = DRGBT603.PR()
npr_dict = DRGBT603.NPR()
sr_dict = DRGBT603.SR()

# print(pr_dict["tracker1"][0])
# print(npr_dict["tracker1"][0])
# print(sr_dict["tracker1"][0])

print(pr_dict["tracker2"][0])
print(npr_dict["tracker2"][0])
print(sr_dict["tracker2"][0])

# lasher.draw_plot(metric_fun=lasher.PR)
# lasher.draw_plot(metric_fun=lasher.NPR)
# lasher.draw_plot(metric_fun=lasher.SR)