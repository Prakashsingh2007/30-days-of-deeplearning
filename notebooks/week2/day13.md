# Day 13 – Introduction to Object Detection

---

# Task 1: What is Object Detection?

**Object detection** is a computer vision task where a model identifies **what objects are present in an image and where they are located**. It predicts both the **class** of each object (such as cat, dog, or car) and a **bounding box** around each object.

For example, if an image contains a cat and a dog, an object detection model might output:

* Cat → Bounding box: `(20, 40, 180, 220)`
* Dog → Bounding box: `(220, 60, 420, 280)`

Unlike image classification, object detection can find **multiple objects** in a single image and determine their positions.

---

# Task 2: Classification vs Detection

| Feature           | Classification                       | Object Detection                                     |
| ----------------- | ------------------------------------ | ---------------------------------------------------- |
| Goal              | Identify what object is in the image | Identify what objects are present and where they are |
| Output            | One or more class labels             | Class labels + bounding boxes                        |
| Number of Objects | Usually one main object              | One or many objects                                  |
| Localization      | ❌ No                                 | ✅ Yes                                                |
| Example           | "This image contains a cat."         | "Cat at (20,40,180,220), Dog at (220,60,420,280)"    |

**Example:**

**Classification**

```
Image
  ↓
Cat
```

**Detection**

```
Image
  ↓
Cat → Box
Dog → Box
```

---

# Task 3: Draw a Simple Image and Bounding Boxes

```
+------------------------------------------------------+
|                                                      |
|  +-----------+                     +--------------+  |
|  |           |                     |              |  |
|  |   CAT     |                     |     DOG      |  |
|  |           |                     |              |  |
|  +-----------+                     +--------------+  |
|                                                      |
+------------------------------------------------------+
```

Example bounding box coordinates:

**Cat**

```
(20, 40, 180, 220)
```

Meaning:

* x_min = 20
* y_min = 40
* x_max = 180
* y_max = 220

---

**Dog**

```
(220, 60, 420, 280)
```

Meaning:

* x_min = 220
* y_min = 60
* x_max = 420
* y_max = 280

---

# Task 4

## What is IoU?

**IoU (Intersection over Union)** measures how much the predicted bounding box overlaps with the ground-truth bounding box.

Formula:

```
IoU =
Area of Overlap
----------------------------
Area of Union
```

Example:

```
Ground Truth
+--------------+
|              |
|   +------+   |
|   |Pred  |   |
|   +------+   |
|              |
+--------------+
```

* Large overlap → High IoU (good prediction)
* Small overlap → Low IoU (poor prediction)

IoU ranges from **0 to 1**:

* 0 → No overlap
* 1 → Perfect overlap

---

## What is NMS?

**NMS (Non-Maximum Suppression)** is a technique used to remove **duplicate bounding boxes** that refer to the same object.

Example:

```
Cat

+------------+
|            |
| +--------+ |
| | Cat    | |
| +--------+ |
|            |
+------------+
```

A detector may produce several overlapping boxes for the same cat.

NMS keeps the box with the **highest confidence score** and removes the others if they overlap too much (based on an IoU threshold).

---

## Why is NMS Needed?

Without NMS:

```
Cat
Box 1
Box 2
Box 3
```

The model may report the same object multiple times.

With NMS:

```
Cat
One final bounding box
```

Benefits:

* Removes duplicate detections
* Produces cleaner predictions
* Improves readability and evaluation accuracy

---

# Task 5: Comparison of Popular Object Detection Models

| Model                          | Speed           | Accuracy        | Best Use Case                                                                                               |
| ------------------------------ | --------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| **YOLO (You Only Look Once)**  | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐⭐ High       | Real-time applications such as autonomous driving, surveillance, robotics, and video analytics              |
| **SSD (Single Shot Detector)** | ⭐⭐⭐⭐ Fast       | ⭐⭐⭐ Good        | Mobile devices, embedded systems, and applications requiring a balance between speed and accuracy           |
| **Faster R-CNN**               | ⭐⭐ Slower       | ⭐⭐⭐⭐⭐ Very High | Medical imaging, research, quality inspection, and applications where accuracy is more important than speed |

### Summary

* **YOLO**: Fastest detector, ideal for real-time inference.
* **SSD**: Good trade-off between speed and accuracy, especially on lightweight hardware.
* **Faster R-CNN**: Highest accuracy but slower, making it suitable when detection quality matters more than inference speed.

---

## Key Takeaways

* **Classification** answers: *"What is in the image?"*
* **Object Detection** answers: *"What objects are in the image, and where are they?"*
* **Bounding boxes** specify an object's location.
* **IoU** measures how well a predicted box matches the true box.
* **NMS** removes duplicate detections of the same object.
* **YOLO**, **SSD**, and **Faster R-CNN** represent different trade-offs between speed and accuracy, with the best choice depending on the application's requirements.
