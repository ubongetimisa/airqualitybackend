"""
Health Assessment Module for Air Quality

This module provides health impact assessment based on PM2.5 values
using WHO and EPA standards.

Author: Ubong Isaiah Eka
Email: ubongisaiahetim001@gmail.com
Date: 2025
"""

from typing import Dict, Any
from pydantic import BaseModel


class HealthImpactAssessment(BaseModel):
    """Health impact assessment result"""
    risk_level: str
    aqi_category: str
    health_implications: str
    recommendations: str
    who_standard: bool = True


def assess_health_impact(pm25: float) -> HealthImpactAssessment:
    """
    Assess health impact based on PM2.5 concentration using WHO standards.
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        HealthImpactAssessment: Health assessment with risk level and recommendations
    """
    
    # WHO Air Quality Guidelines (2021) - PM2.5
    if pm25 <= 12:
        return HealthImpactAssessment(
            risk_level="Good",
            aqi_category="0-12 µg/m³",
            health_implications="Air quality is satisfactory.",
            recommendations="No action needed.",
            who_standard=True
        )
    elif pm25 <= 35.4:
        return HealthImpactAssessment(
            risk_level="Moderate",
            aqi_category="12-35.4 µg/m³",
            health_implications="Acceptable air quality. Unusually sensitive groups may experience health issues.",
            recommendations="Sensitive groups should reduce prolonged outdoor activities.",
            who_standard=True
        )
    elif pm25 <= 55.4:
        return HealthImpactAssessment(
            risk_level="Unhealthy for Sensitive Groups",
            aqi_category="35.4-55.4 µg/m³",
            health_implications="Members of general public are beginning to experience health effects.",
            recommendations="Sensitive groups should avoid outdoor activities. General public should reduce outdoor time.",
            who_standard=True
        )
    elif pm25 <= 150.4:
        return HealthImpactAssessment(
            risk_level="Unhealthy",
            aqi_category="55.4-150.4 µg/m³",
            health_implications="Everyone may begin to experience health effects. Levels may be hazardous.",
            recommendations="Sensitive groups should avoid outdoor activities. General public should reduce prolonged outdoor activities.",
            who_standard=True
        )
    else:
        return HealthImpactAssessment(
            risk_level="Hazardous",
            aqi_category=">150.4 µg/m³",
            health_implications="Health alert: The entire population is more likely to be affected. Serious health effects possible.",
            recommendations="Everyone should avoid outdoor activities. Wear high-quality masks if outdoors is necessary. Use air purifiers indoors.",
            who_standard=True
        )


def get_aqi_color(pm25: float) -> str:
    """
    Get AQI color code based on PM2.5 value.
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        str: Hex color code for visualization
    """
    if pm25 <= 12:
        return "#00E400"  # Green
    elif pm25 <= 35.4:
        return "#FFFF00"  # Yellow
    elif pm25 <= 55.4:
        return "#FF7E00"  # Orange
    elif pm25 <= 150.4:
        return "#FF0000"  # Red
    else:
        return "#8F3F97"  # Purple


def get_health_recommendations(pm25: float, age_group: str = "general") -> Dict[str, Any]:
    """
    Get detailed health recommendations based on PM2.5 and age group.
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        age_group: "children", "elderly", "general"
        
    Returns:
        Dict: Detailed recommendations
    """
    
    assessment = assess_health_impact(pm25)
    
    recommendations = {
        "risk_level": assessment.risk_level,
        "pm25_value": round(pm25, 2),
        "outdoor_activities": [],
        "indoor_precautions": [],
        "special_groups": [],
        "general_public": []
    }
    
    if pm25 <= 12:
        recommendations["outdoor_activities"] = ["All activities allowed"]
        recommendations["indoor_precautions"] = ["No special precautions needed"]
        recommendations["general_public"] = ["Enjoy outdoor activities normally"]
        
    elif pm25 <= 35.4:
        recommendations["outdoor_activities"] = ["Outdoor activities allowed", "Sensitive groups should limit intense outdoor exercise"]
        recommendations["indoor_precautions"] = ["Keep windows closed on high pollution days"]
        if age_group == "children":
            recommendations["special_groups"] = ["Children: Reduce intense outdoor play"]
        elif age_group == "elderly":
            recommendations["special_groups"] = ["Elderly: Reduce intense physical activities"]
            
    elif pm25 <= 55.4:
        recommendations["outdoor_activities"] = ["Limit outdoor activities", "Avoid strenuous activities outdoors"]
        recommendations["indoor_precautions"] = ["Keep windows and doors closed", "Use air conditioning with filters"]
        if age_group == "children":
            recommendations["special_groups"] = ["Children: Keep indoors or minimize outdoor time"]
        elif age_group == "elderly":
            recommendations["special_groups"] = ["Elderly: Stay indoors as much as possible"]
            
    elif pm25 <= 150.4:
        recommendations["outdoor_activities"] = ["Minimize outdoor activities", "Wear N95/KN95 masks if outdoors is necessary"]
        recommendations["indoor_precautions"] = ["Keep all windows and doors closed", "Use HEPA air purifiers", "Seal window gaps"]
        if age_group == "children":
            recommendations["special_groups"] = ["Children: Keep indoors", "Monitor for respiratory symptoms"]
        elif age_group == "elderly":
            recommendations["special_groups"] = ["Elderly: Stay indoors", "Take prescribed respiratory medications"]
            
    else:
        recommendations["outdoor_activities"] = ["Avoid all outdoor activities", "Mandatory mask use if outdoors"]
        recommendations["indoor_precautions"] = ["Keep all windows and doors tightly closed", "Use multiple HEPA air purifiers", "Seal all gaps and cracks"]
        if age_group == "children":
            recommendations["special_groups"] = ["Children: Keep indoors in controlled environment", "Seek medical attention if breathing difficulties"]
        elif age_group == "elderly":
            recommendations["special_groups"] = ["Elderly: Keep indoors", "Consult doctor immediately for any symptoms"]
    
    return recommendations
