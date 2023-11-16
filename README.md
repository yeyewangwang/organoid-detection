# Organoid Detection

In May 2022, Lizzie Kumar, Alex Wong, and I trimmed Yolo V3. We showed that training from scratch doesn't work.

### Work in Progress, Image Detection Evolved:
1. Basic techniques
    1. Based on the dataset, and the canny edges, organoids are circles with deep black countours (say, intensity < 64) that are 3-5 pixels wide. Some could have rough patterns in the center; most don't.
    ![alt text](gallery/canny_edges.png)
    ![alt text](gallery/canny_edges_2.png)
    <!-- 2. Ask CoPilot, "How do I filter out high intensity circles using OpenCV?". CoPilot recommends Gaussian blur and Hough Circles. -->
2. Minimum viable product: Fine-tune a pretrained model using HuggingFace
    1. Produce metadata.jsonl for HuggingFace default loading script.
3. Use a multi-modal LLM API
4. Training using only Canny edges and bounding boxes
