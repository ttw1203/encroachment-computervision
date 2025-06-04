from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir="D:\\thesisVideo\\weights\\BNS.v5i.coco", epochs=15, batch_size=1, grad_accum_steps=1, lr=1e-4, output_dir="D:\\thesisVideo\\weights")