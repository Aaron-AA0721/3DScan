using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit;

public class PointCloudManipulator : MonoBehaviour
{
    public XRInteractionManager xRInteractionManager;
    PointCloudRenderer PCDRenderer;
    Fusion fusion;
    public Transform MagicLeapController; 
    ActionBasedController  xrController;
    public GameObject DeleteCylinder;
    // float deleteRadius = 0.05f;
    public Slider RadiusSlider;
    public TextMeshProUGUI RadiusText;
    bool DeletionMode = false;
    public TextMeshProUGUI DeletionModeButton;
    bool DeleteButtonPressed = false;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        PCDRenderer = FindFirstObjectByType<PointCloudRenderer>();
        xrController = MagicLeapController.GetComponent<ActionBasedController>();
        fusion = FindFirstObjectByType<Fusion>();
        DeleteCylinder.SetActive(DeletionMode);
    }
    public void SwitchDeletionMode()
    {
        DeletionMode = !DeletionMode;
        DeletionModeButton.text = DeletionMode ? "Deletion On" : "Deletion Off";
        DeleteCylinder.SetActive(DeletionMode);
    }
    public void IncreaseRadius()
    {
        RadiusSlider.value += 0.2f;
        if (RadiusSlider.value > RadiusSlider.maxValue)
            RadiusSlider.value = RadiusSlider.maxValue;
    }
    public void DecreaeseRadius()
    {
        RadiusSlider.value -= 0.2f;
        if (RadiusSlider.value < RadiusSlider.minValue)
            RadiusSlider.value = RadiusSlider.minValue;
    }
    // Update is called once per frame
    void Update()
    {
        // Debug.Log(MagicLeapController.position);
        // Debug.Log(MagicLeapController.forward);
        DeleteCylinder.transform.localScale = new Vector3(RadiusSlider.value * 2 / 100, 1, RadiusSlider.value * 2 / 100);
        RadiusText.text = $"Radius: {RadiusSlider.value} cm";
        if (xrController != null && xrController.selectAction.action != null)
        {
            if (xrController.selectAction.action.ReadValue<float>() > 0.5f && DeletionMode && !DeleteButtonPressed)
            {
                DeleteButtonPressed = true;
                Debug.Log("Trigger is pressed!");
                var ray = new Ray(MagicLeapController.position, MagicLeapController.forward);
                // Debug.Log("PCDRenderer.GetMeshColliders(): " + PCDRenderer.GetMeshColliders().Count);
                fusion.DeletePointByRay(ray, RadiusSlider.value / 100);
                Debug.Log("DeletionFinished");
            }
            if(xrController.selectAction.action.ReadValue<float>() < 0.5f)
            {
                DeleteButtonPressed = false;
            }
        }
        else
        {
            xrController = MagicLeapController.GetComponent<ActionBasedController>();
        }
    }
    
}
