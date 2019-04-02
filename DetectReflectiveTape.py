import libjevois as jevois
import cv2
import numpy as np
import math
import json

class DetectReflectiveTape:

    # ##########################################################################
    ## Constructor
    def __init__(self):
        # HSV Color Parameters
        self.hsv_hue = [60.0, 100.0]
        self.hsv_saturation = [175.0, 255.0]
        self.hsv_value = [175.0, 255.0]

        # Erode Parameters
        self.erode_kernal = None
        self.erode_anchor = (-1, -1)
        self.erode_iterations = 1.0
        self.erode_bordertype = cv2.BORDER_CONSTANT
        self.erode_bordervalue = (-1)

        # Filter Contour Parameters (Pixels)
        self.min_area = 300

        # Camera Parameters (Inches)
        self.image_width = 640
        self.image_height = 480
        self.focal_length = 696.195

        # Manipulator constants (Inches)
        self.hatch_intake_offset = 0
        self.cargo_intake_offset = 0

        # Physical vision target constants
        self.target_width = 11.0630053
        self.tape_width = 2
        self.actual_depth = None
        self.offset_angle = None

        # END CONSTANTS
    # ##########################################################################

    ## Process function with serial output
    def process(self, inframe, outframe):
    # ##########################################################################

        # BEGIN GRIP CODE

    # ##########################################################################

        # Get raw image from Jevois
        raw_image = inframe.getCvBGR()

        # Applies HSV filter to image
        hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        threshold_hsv_image = cv2.inRange(hsv_image, (self.hsv_hue[0], self.hsv_saturation[0], self.hsv_value[0]),  (self.hsv_hue[1], self.hsv_saturation[1], self.hsv_value[1]))

        # Erode Image
        #eroded_image = cv2.erode(threshold_hsv_image, self.erode_kernal, self.erode_iterations)

        # Find contours
        contours, hierarchy = cv2.findContours(threshold_hsv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

        # Filter contours
        filtered_contours = []
        for contour in contours:
            if (contours != None) and (len(contours) > 0):
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    filtered_contours.append(contour)

    # ##########################################################################

        # END GRIP CODE

    # ##########################################################################

        ## Begin image processing

        # Creates bounding box for all contours
        boxes = []
        for contour in filtered_contours:
            rect = cv2.minAreaRect(contour)
            boxes.append(rect)

            # Draws all contours to output image in red
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(raw_image, [box], 0, (0, 0, 255), 2)

            """centerX = rect[0][0]
            centerY = rect[0][1]
            pixel_width = rect[1][0]
            pixel_height = rect[1][1]
            rect_angle = rect[2]
            # Switch Width and Height
            if pixel_width > pixel_height:
                temp = pixel_height
                pixel_height = pixel_width
                pixel_width = temp"""

        # Gets target boxes
        sorted_boxes = self.sort(boxes)
        box_types = self.get_box_types(sorted_boxes)

        if len(sorted_boxes) >= 2 and len(box_types) >= 2 and len(sorted_boxes) == len(box_types):
            # Gets target boxes out of visible boxes
            target_boxes = self.get_target_boxes(sorted_boxes, box_types)

            if target_boxes != None:
                # Draws target boxes in blue
                for target_box in target_boxes:
                    box = cv2.boxPoints(target_box)
                    box = np.int0(box)
                    cv2.drawContours(raw_image, [box], 0, (255, 0, 0), 2)

                # Collects information from target boxes
                centerX1 = target_boxes[0][0][0]
                centerX2 = target_boxes[1][0][0]
                centroid_x = self.get_centroid_x(centerX1, centerX2)
                centroid_offset_x = self.get_centroid_offset_x(centerX1, centerX2)

                # Calculates distance from target boxes
                subtended_angle = self.get_subtended_angle(centerX1, centerX2)
                target_distance = self.get_depth_with_angle(subtended_angle)

                # Calculates offset angle of center of target to center of camera view
                offset_angle = self.get_offset_angle(centroid_x)

                # Outputs data to RoboRIO
                vision_data = {"Target_Center_X": centroid_x, "Distance": target_distance, "Angle": offset_angle}
                json_vision_data = json.dumps(vision_data)
                jevois.sendSerial(json_vision_data)
        else:
            vision_data = {"Error": "No targets found."}
            json_vision_data = json.dumps(vision_data)
            jevois.sendSerial(json_vision_data)

        # Done using Jevois camera
        inframe.done()

        # Output HSV image
        output_image = cv2.cvtColor(threshold_hsv_image, cv2.COLOR_GRAY2BGR)
        outframe.sendCvBGR(raw_image)

    # ##########################################################################

        # END GRIP CODE

    # ##########################################################################

    # Sorts boxes based on x-position
    def sort(self, boxes):
        if len(boxes) >= 2:
            sorted_boxes = []

            while len(boxes) > 0:
                min_index = 0
                min_value = boxes[0][0][0]

                for i in range(len(boxes)):
                    if boxes[i][0][0] < min_value:
                        min_index = i
                        min_value = boxes[i][0][0]

                sorted_boxes.append(boxes[min_index])
                boxes.pop(min_index)

            return sorted_boxes

        return boxes

    # Stores boxes based on "box type", which is either a 0 or 1, depending on angle of box
    def get_box_types(self, sorted_boxes):
        box_types = []

        for box in sorted_boxes:
            rect_angle = abs(box[2])

            # Left Box
            if rect_angle >= 65.5 and rect_angle <= 85.0:
                box_types.append(0)
            # Right Box
            elif rect_angle >= 4.5 and rect_angle <= 24.5:
                box_types.append(1)

        return box_types

    def get_target_boxes(self, sorted_boxes, box_types):
        # If first box is the right of target, delete from array
        if box_types[0] == 1:
            sorted_boxes.pop(0)
            box_types.pop(0)

        # If last box is the left of a target, delete from array
        last_element = len(sorted_boxes) - 1
        if box_types[last_element] == 0:
            sorted_boxes.pop(last_element)
            box_types.pop(last_element)

        for i in range(0, len(box_types) - 1):
            # If two subsequent boxes are 0, delete the left box
            if box_types[i] == 0 and box_types[i + 1] == 0:
                sorted_boxes.pop(i)
                box_types.pop(i)
                i -= 1

        # If two subsequent boxes are 1, delete the right box
        for i in range(0, len(box_types) - 1):
            if box_types[i] == 1 and box_types[i + 1] == 1:
                sorted_boxes.pop(i + 1)

                box_types.pop(i + 1)

        if len(sorted_boxes) > 2:
            center = self.image_width / 2

            centerX1 = sorted_boxes[0][0][0]
            centerX2 = sorted_boxes[1][0][0]
            centroid_x = self.get_centroid_x(centerX1, centerX2)

            min_pixel_distance = abs(center - centroid_x)
            min_distance_index = 0

            for i in range(2, len(sorted_boxes), 2):
                centerX1 = sorted_boxes[i][0][0]
                centerX2 = sorted_boxes[i + 1][0][0]

                centroid_x = self.get_centroid_x(centerX1, centerX2)
                pixel_distance = abs(center - centroid_x)

                actual_avg_width = self.get_physical_width_between_tape(sorted_boxes[i], sorted_boxes[i + 1])

                if pixel_distance < min_pixel_distance and actual_avg_width >= 8.5630053 and actual_avg_width <= 13.5630053:
                    min_pixel_distance = pixel_distance
                    min_distance_index = i

            target_boxes = [sorted_boxes[min_distance_index], sorted_boxes[min_distance_index + 1]]

            actual_avg_width = self.get_physical_width_between_tape(target_boxes[0], target_boxes[1])

            if actual_avg_width >= 8.5630053 and actual_avg_width <= 13.5630053:
                return target_boxes

            return None
        elif len(sorted_boxes) == 2:
            actual_avg_width = self.get_physical_width_between_tape(sorted_boxes[0], sorted_boxes[1])

            if actual_avg_width >= 8.5630053 and actual_avg_width <= 13.5630053:
                return sorted_boxes

            return None

        return None

    # Distance to Object = Focal Length * Actual Width of Object / Pixel Width of Object (Less accurate method, only for checking of valid target)
    def get_depth_with_width(self, pixel_width):
        ## Returns distance to target in inches
        return self.focal_length * self.tape_width / pixel_width

    # Gets the subtended angle between positions left_x and right_x on image
    def get_subtended_angle(self, left_x, right_x):
        ## Returns angle in degrees
        return abs(math.atan((left_x - self.image_width / 2) / self.focal_length) - math.atan((right_x - self.image_width / 2) / self.focal_length))

    # Distance to Object = Actual Width of Object / (2 * tan(Subtended Angle / 2))
    def get_depth_with_angle(self, subtended_angle):
        ## Returns distance to object in inches
        return self.target_width / (2 * math.tan(subtended_angle / 2))

    # Actual Width of Object = Pixel Width of Object * Distance to Object / Focal Length
    def get_physical_width_between_tape(self, box1, box2):
        ## Gets width of both boxes
        pixel_width1 = box1[1][0]
        pixel_width2 = box2[1][0]

        ## Gets height of both boxes
        pixel_height1 = box1[1][1]
        pixel_height2 = box2[1][1]

        ## Gets the center x-position of both boxes
        centerX1 = box1[0][0]
        centerX2 = box2[0][0]

        ## Checks if the width is greater than the height, if it is, switch them
        if pixel_width1 > pixel_height1:
            temp = pixel_height1
            pixel_height1 = pixel_width1
            pixel_width1 = temp

        if pixel_width2 > pixel_height2:
            temp = pixel_height2
            pixel_height2 = pixel_width2
            pixel_width2 = temp

        ## Calculates the average width of the two reflective tapes in pixels
        avg_pixel_width = (pixel_width1 + pixel_width2) / 2

        ## Calculates distance to tapes
        actual_depth = self.get_depth_with_width(avg_pixel_width)
        pixel_width_between_boxes = self.get_pixel_width(centerX1, centerX2)

        ## Returns width between center of tapes in inches
        return pixel_width_between_boxes * actual_depth / self.focal_length

    # Offset Angle = arctan((Offset Position - (Image Width / 2)) / Focal Length)
    def get_offset_angle(self, offset_position):
        ## Returns an angle in degrees
        return math.degrees(math.atan((offset_position - (self.image_width / 2)) / self.focal_length))

    ## Returns the average of two values
    def get_centroid_x(self, centerX1, centerX2):
        return (centerX2 + centerX1) / 2

    # Returns the distance between two values
    def get_pixel_width(self, centerX1, centerX2):
        return centerX2 - centerX1

    # Gets centroid while accounting for intake offset
    def get_centroid_offset_x(self, centerX1, centerX2):
        pixel_width = self.get_pixel_width(centerX1, centerX2)

        return centerX2 - pixel_width * ((self.target_width) / 2 - self.hatch_intake_offset) / self.target_width
