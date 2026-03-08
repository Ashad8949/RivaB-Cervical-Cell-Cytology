import timm
try:
    m = timm.create_model("convnextv2_base", pretrained=False)
    print(f"Type of feature_info: {type(m.feature_info)}")
    print(f"Content of feature_info: {m.feature_info}")
    if isinstance(m.feature_info, list):
         print("Element 0 type:", type(m.feature_info[0]))
         print("Element 0 content:", m.feature_info[0])
except Exception as e:
    print(e)
