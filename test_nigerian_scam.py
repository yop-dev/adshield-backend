#!/usr/bin/env python3
"""
Test Nigerian prince/advance-fee scam detection
"""

import asyncio
import httpx
import json

async def test_nigerian_scam():
    """Test with classic Nigerian prince scam text"""
    
    # Classic Nigerian prince scam email text
    scam_text = """
    Dear Beloved Friend,
    
    I am Barrister John Smith, attorney to late Mr. James Williams who died in a 
    plane crash leaving behind $45 million dollars in a Nigerian bank account.
    
    I need your assistance to transfer this inheritance money to your bank account 
    as the beneficiary. You will receive 40% of the total sum for your help.
    
    This is 100% legitimate and risk free. Please send me your bank account details 
    and a processing fee of $5000 to release the funds.
    
    This is confidential. Reply urgently.
    
    God bless,
    Barrister John Smith
    """
    
    url = "http://localhost:8000/api/v1/text/analyze"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"text": scam_text},
            timeout=30
        )
        
        print("Nigerian Prince Scam Test")
        print("=" * 50)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nDetection Result: {result.get('label', 'unknown').upper()}")
            print(f"Confidence Score: {(result.get('score', 0) * 100):.1f}%")
            
            if result.get('reasons'):
                print("\nReasons Detected:")
                for reason in result['reasons']:
                    print(f"  • {reason}")
            
            if result.get('highlights'):
                print(f"\nSuspicious phrases found: {len(result['highlights'])}")
            
            # Check if it was properly detected
            if result.get('label') == 'phishing' and result.get('score', 0) > 0.8:
                print("\n✅ SUCCESS: Nigerian prince scam properly detected!")
            else:
                print("\n❌ FAILURE: Scam not properly detected")
                print(f"   Expected: phishing with >80% confidence")
                print(f"   Got: {result.get('label')} with {(result.get('score', 0) * 100):.1f}% confidence")
        else:
            print(f"Error: {response.text}")

async def test_advance_fee_variations():
    """Test various advance-fee scam patterns"""
    
    test_cases = [
        {
            "name": "Lottery Scam",
            "text": "Congratulations! You've won $10 million in the international lottery. Send $500 processing fee to claim your prize."
        },
        {
            "name": "Inheritance Scam", 
            "text": "Dear Sir, You are the beneficiary of a $25 million inheritance from a deceased relative in Nigeria. Contact our attorney immediately."
        },
        {
            "name": "Business Proposal",
            "text": "I have a business proposal worth 15 million USD. I need a foreign partner to help transfer these trapped funds. You will get 30%."
        },
        {
            "name": "Legitimate Email (Control)",
            "text": "Hi John, Here's the quarterly report you requested. The meeting is scheduled for Tuesday at 2 PM. Best regards, Sarah"
        }
    ]
    
    url = "http://localhost:8000/api/v1/text/analyze"
    
    print("\n\nAdvance-Fee Scam Variations Test")
    print("=" * 50)
    
    for test in test_cases:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"text": test["text"]}, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                is_scam = result.get('label') == 'phishing'
                confidence = result.get('score', 0) * 100
                
                print(f"\n{test['name']}:")
                print(f"  Result: {'SCAM' if is_scam else 'SAFE'} ({confidence:.1f}% confidence)")
                
                # Check if detection is correct
                if "Legitimate" in test['name']:
                    if not is_scam:
                        print(f"  ✅ Correctly identified as legitimate")
                    else:
                        print(f"  ❌ False positive - marked legitimate email as scam")
                else:
                    if is_scam and confidence > 70:
                        print(f"  ✅ Correctly identified as scam")
                    else:
                        print(f"  ❌ Failed to detect scam")

async def main():
    print("Testing Advance-Fee/Nigerian Prince Scam Detection\n")
    
    # Wait for backend to start
    await asyncio.sleep(3)
    
    await test_nigerian_scam()
    await test_advance_fee_variations()

if __name__ == "__main__":
    asyncio.run(main())
