# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:18:50 2020
Get boundaries centerline
@author: cjzhang
"""
import math
import matplotlib.pyplot as plt
def getBoundariesCenterPoints(boundary1, boundary2):
    boundaryPointsNumber1 = len(boundary1)
    boundaryPointsNumber2 = len(boundary2)
    midPointsList1 = []
    midPointsList2 = []
    for i in range (boundaryPointsNumber1):
        x1 = boundary1[i][0]
        y1 = boundary1[i][1]
        minDistance = 1000000
        for j in range(boundaryPointsNumber2):
            x2 = boundary2[j][0]
            y2 = boundary2[j][1]
            tempDistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (tempDistance < minDistance):
                minDistance = tempDistance
                nearestPointX = x2
                nearestPointY = y2
        midPointX = (x1 + nearestPointX) / 2
        midPointY = (y1 + nearestPointY) / 2
        midPointsList1.append([midPointX, midPointY])
    for i in range (boundaryPointsNumber2):
        x2 = boundary2[i][0]
        y2 = boundary2[i][1]
        minDistance = 1000000
        for j in range(boundaryPointsNumber1):
            x1 = boundary1[j][0]
            y1 = boundary1[j][1]
            tempDistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (tempDistance < minDistance):
                minDistance = tempDistance
                nearestPointX = x1
                nearestPointY = y1
        midPointX = (x2 + nearestPointX) / 2
        midPointY = (y2 + nearestPointY) / 2
        midPointsList2.append([midPointX, midPointY])
    return midPointsList1, midPointsList2

def getClosestPoint(curPointPositionX, curPointPositionY, curPointPositionYaw, pointList):
    minDistance = 100000
    closestMidPointX = -1
    closestMidPointY = -1
    for i in range(len(pointList)):
        vehicle2MidPoint = [pointList[i][0] - curPointPositionX, pointList[i][1] - curPointPositionY]
        cosAngel = (math.cos(curPointPositionYaw)*(vehicle2MidPoint[0]) + math.sin(curPointPositionYaw)*(vehicle2MidPoint[1]))/(math.sqrt((vehicle2MidPoint[0])*(vehicle2MidPoint[0]) + (vehicle2MidPoint[1])*(vehicle2MidPoint[1])))
        tempDistance = math.sqrt((pointList[i][0] - curPointPositionX)**2 + (pointList[i][1] - curPointPositionY)**2)
        if tempDistance < minDistance and cosAngel > 0.5:
            minDistance = tempDistance
            closestMidPointX = pointList[i][0]
            closestMidPointY = pointList[i][1]
    return closestMidPointX, closestMidPointY

def getBoundariesCenterLinePoints(midPointsList1, midPointsList2, curX, curY, curYaw):
    preMassMidPointsList = []
    for i in range(len(midPointsList1)):
        preMassMidPointsList.append(midPointsList1[i])
    for i in range(len(midPointsList2)):
        preMassMidPointsList.append(midPointsList2[i])
    massMidPointsList = []
    for i in range (len(preMassMidPointsList)):
        if preMassMidPointsList[i] not in massMidPointsList:
            massMidPointsList.append(preMassMidPointsList[i])
        else:
            pass
    centerLine = []
    curPointPositionX = curX
    curPointPositionY = curY
    curPointPositionYaw = curYaw
    pointList = []
    for i in range(len(massMidPointsList)):
        pointList.append(massMidPointsList[i])
    while(1):
        closestMidPointX, closestMidPointY = getClosestPoint(curPointPositionX, curPointPositionY, curPointPositionYaw, pointList)
        if (closestMidPointX == -1 and closestMidPointY == -1):
            break
        centerLine.append([closestMidPointX, closestMidPointY])
        pointList.remove([closestMidPointX, closestMidPointY])
        curPointPositionYaw = math.atan2(closestMidPointY - curPointPositionY, closestMidPointX - curPointPositionX)
        curPointPositionX = closestMidPointX
        curPointPositionY = closestMidPointY
    
    return centerLine

def getBoundariesCenterLine(boundary1, boundary2, curX, curY, curYaw):
    boundary1X, boundary1Y, boundary2X, boundary2Y = [], [], [], []
    for i in range(len(boundary1)):
        boundary1X.append(boundary1[i][0])
        boundary1Y.append(boundary1[i][1])
    for i in range(len(boundary2)):
        boundary2X.append(boundary2[i][0])
        boundary2Y.append(boundary2[i][1])
#    plt.plot(boundary1X, boundary1Y, 'r')
#    plt.plot(boundary2X, boundary2Y, 'r')
    
    midPointsList1, midPointsList2 = getBoundariesCenterPoints(boundary1, boundary2)
    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(midPointsList1)):
        x1.append(midPointsList1[i][0])
        y1.append(midPointsList1[i][1])
    for i in range(len(midPointsList2)):
        x2.append(midPointsList2[i][0])
        y2.append(midPointsList2[i][1])
    plt.plot(x1, y1, "ob", markersize = 3)
    plt.plot(x2, y2, "og", markersize = 3)
    centerLine = getBoundariesCenterLinePoints(midPointsList1, midPointsList2, curX, curY, curYaw)
    centerLineX, centerLineY = [], []
    for i in range(len(centerLine)):
        centerLineX.append(centerLine[i][0])
        centerLineY.append(centerLine[i][1])
#    plt.plot(centerLineX, centerLineY, 'black')
#    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
#    plt.legend(["boundary1", "boundary2", "massPoints1", "massPoints2", "centerLine"], prop=font1)
#    plt.show()
    return centerLineX, centerLineY
    
            
            
