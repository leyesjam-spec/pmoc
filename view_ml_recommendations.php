<?php
/**
 * View AI Recommendations
 * Detailed view of AI-generated counseling recommendations
 */

require_once '../includes/session.php';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>View Counseling Topics Recommendations - BCPDO System</title>
    <?php include '../includes/header.php'; ?>
</head>

<body class="hold-transition sidebar-mini layout-fixed">
    <div class="wrapper">
        <?php include '../includes/navbar.php'; ?>
        <?php include '../includes/sidebar.php'; ?>
        
        <div class="content-wrapper">
  <section class="content-header">
    <div class="container-fluid">
      <div class="row mb-2">
        <div class="col-sm-6">
          <h1>Counseling Topics Recommendations</h1>
        </div>
      </div>
    </div>
  </section>

  <section class="content">
    <div class="container-fluid">
      <div class="row">
        <div class="col-12">
          <div class="card">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-center mb-2">
                <h5 id="detailTitle" class="mb-0">Counseling Topics Recommendations</h5>
                <a href="./ml_dashboard.php" class="btn btn-sm btn-secondary">Back to Dashboard</a>
              </div>
              <div id="result" class="mb-2" style="white-space: pre-wrap;"></div>
              <div class="mb-3" id="summaryWrap" style="display:none;">
                <h6 class="text-muted mb-2"><i class="fas fa-lightbulb mr-1"></i>Counseling Topics Analysis Summary</h6>
                <blockquote class="blockquote pl-3 border-left border-info" id="aiSummary"></blockquote>
              </div>
              <div id="cardsWrap"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

        <?php include '../includes/footer.php'; ?>
    </div>

    <?php include '../includes/scripts.php'; ?>

<script>
(function() {
  const result = document.getElementById('result');
  const detailTitle = document.getElementById('detailTitle');
  const aiSummary = document.getElementById('aiSummary');
  const summaryWrap = document.getElementById('summaryWrap');
  const cardsWrap = document.getElementById('cardsWrap');

  function badge(level){
    const map = { High: 'danger', Medium: 'warning', Low: 'success' };
    const cls = map[level] || 'secondary';
    return '<span class="badge badge-' + cls + '">' + level + '</span>';
  }

  async function fetchCouples() {
    const resp = await fetch('../couple_list/get_couples.php', { credentials: 'same-origin' });
    const data = await resp.json();
    if (!data.success) throw new Error('Failed to load couples');
    return data.data;
  }

  async function getAnalysisResults(accessId) {
    const resp = await fetch('./ml_api.php?action=get_analysis&access_id=' + accessId, { credentials: 'same-origin' });
    const data = await resp.json();
    if (data.status !== 'success') throw new Error(data.message || 'Failed to fetch analysis');
    // Check if couple has been analyzed
    if (!data.analyzed) {
      throw new Error('This couple has not been analyzed yet. Please run "Analyze All Couples" from the ML Dashboard first.');
    }
    return data;
  }

  async function renderDetail(accessId) {
    try {
      const couples = await fetchCouples();
      const couple = couples.find(x => String(x.access_id) === String(accessId));
      if (couple) detailTitle.textContent = 'Counseling Topics Recommendations for ' + couple.couple_names;

      const analysis = await getAnalysisResults(accessId);
      
      // Display risk level and summary
      const riskLevel = analysis.risk_level;  // Overall hybrid risk level
      const mlRecommendations = analysis.recommendations || [];
      const focusCategories = analysis.focus_categories || [];
      const mlConfidence = analysis.ml_confidence || 0;
      const riskReasoning = analysis.risk_reasoning || '';
      const counselingReasoning = analysis.counseling_reasoning || '';
      
      console.log('Analysis data:', analysis);
      console.log('ML Recommendations:', mlRecommendations);
      console.log('Focus Categories:', focusCategories);
      console.log('Total categories received:', focusCategories.length);
      console.log('All category names:', focusCategories.map(c => c.name));
      
      // Helper function to generate dynamic analysis text
      const getDynamicAnalysisText = (categories, risk, confidence) => {
        const count = categories.length;
        const criticalCount = categories.filter(c => c.priority === 'Critical').length;
        const highCount = categories.filter(c => c.priority === 'High').length;
        
        let text = '';
        
        // Opening based on count
        if (count === 4) {
          text = 'Identified <strong class="text-primary">all 4 MEAI categories</strong> requiring counseling attention';
        } else if (count === 3) {
          text = 'Identified <strong class="text-primary">3 out of 4 MEAI categories</strong> requiring counseling attention';
        } else if (count === 2) {
          text = 'Identified <strong class="text-primary">2 MEAI categories</strong> requiring counseling attention';
        } else if (count === 1) {
          text = 'Identified <strong class="text-primary">1 MEAI category</strong> requiring counseling attention';
        } else {
          text = 'No priority categories identified - couple appears to be doing well';
        }
        
        // Add priority details if categories exist
        if (count > 0) {
          if (criticalCount > 0) {
            text += `, including <strong class="text-danger">${criticalCount} critical priority area${criticalCount > 1 ? 's' : ''}</strong>`;
          } else if (highCount === count) {
            text += ', all marked as <strong class="text-warning">high priority</strong>';
          } else if (highCount > 0) {
            text += `, with <strong class="text-warning">${highCount} high priority area${highCount > 1 ? 's' : ''}</strong>`;
          }
        }
        
        // Add recommendation based on risk and confidence - consistent with counseling intensity
        if (count > 0) {
          if (risk === 'High' && confidence > 0.6) {
            text += '. <strong>Immediate comprehensive counseling strongly recommended.</strong>';
          } else if (risk === 'High' || confidence > 0.6) {
            text += '. <strong>Comprehensive counseling sessions recommended.</strong>';
          } else if (confidence > 0.3) {
            text += '. <strong>Structured counseling sessions recommended.</strong>';
          } else {
            text += '. <strong>Preventive counseling recommended.</strong>';
          }
        }
        
        return text;
      };
      
      // Use actual personalized recommendations from ML analysis
      const getPersonalizedRecommendations = (mlRecommendations, focusCategories) => {
        const recommendations = [];
        
        // Use the actual personalized recommendations from the ML analysis
        if (mlRecommendations && mlRecommendations.length > 0) {
          // Display the personalized recommendations, filtering out unwanted messages
          mlRecommendations.forEach(rec => {
            // Filter out the "balanced partnership dynamics" message
            if (rec.includes('balanced partnership dynamics') && rec.includes('mutual respect and equal voice')) {
              return; // Skip this recommendation
            }
            recommendations.push(rec);
          });
        } else {
          // Fallback if no personalized recommendations available
          recommendations.push('⚠️ No personalized recommendations available - please re-analyze this couple');
        }
        
        // Add category-specific explanations
        if (focusCategories && focusCategories.length > 0) {
          // Get all 4 MEAI categories to show which ones are/aren't predicted
          const allCategories = [
            'Marriage And Relationship',
            'Responsible Parenthood', 
            'Planning The Family',
            'Maternal Neonatal Child Health And Nutrition'
          ];
          
          const predictedCategories = focusCategories.map(c => c.name);
          const notPredictedCategories = allCategories.filter(cat => !predictedCategories.includes(cat));
          
          if (notPredictedCategories.length > 0) {
            recommendations.push(`<strong>📋 Category Analysis:</strong> No specific recommendations for ${notPredictedCategories.join(', ')} - these areas show low priority scores (below 20%) indicating the couple is doing well in these areas.`);
          }
        }
        
        return recommendations;
      };
      
      // Create enhanced summary with better readability
      // Calculate priority based on risk level and category scores, not ML confidence
      let priorityPercentage = 0;
      if (riskLevel === 'High') {
        priorityPercentage = 85; // High risk = High priority
      } else if (riskLevel === 'Medium') {
        priorityPercentage = 60; // Medium risk = Medium priority
      } else {
        priorityPercentage = 25; // Low risk = Low priority
      }
      
      // Adjust based on highest category score
      if (focusCategories && focusCategories.length > 0) {
        const maxCategoryScore = Math.max(...focusCategories.map(cat => cat.score));
        const categoryAdjustment = maxCategoryScore * 20; // Scale category score to 0-20%
        priorityPercentage = Math.min(95, priorityPercentage + categoryAdjustment);
      }
      
      const confidencePercentage = priorityPercentage.toFixed(1);
      
      // Get relationship health assessment details with specific reasoning
      let riskIcon = '';
      let riskText = '';
      let riskColor = '';
      
      if (riskLevel === 'High') {
        riskIcon = 'fa-exclamation-circle';
        riskText = 'Significant relationship challenges requiring immediate attention';
        riskColor = 'danger';
      } else if (riskLevel === 'Medium') {
        riskIcon = 'fa-exclamation-triangle';
        riskText = 'Some relationship concerns that need proactive attention';
        riskColor = 'warning';
      } else if (riskLevel === 'Low') {
        riskIcon = 'fa-check-circle';
        riskText = 'Healthy relationship foundation with good communication patterns';
        riskColor = 'success';
      }
      
      // Get counseling recommendation details with specific reasoning
      let confidenceIcon = '';
      let confidenceText = '';
      let confidenceColor = '';
      
      // Set confidence text and color based on priority percentage, not ML confidence
      if (priorityPercentage > 70) {
        confidenceIcon = 'fa-user-md';
        confidenceText = 'Intensive counseling program recommended for comprehensive support';
        confidenceColor = 'danger';
      } else if (priorityPercentage > 40) {
        confidenceIcon = 'fa-handshake';
        confidenceText = 'Structured counseling sessions recommended for relationship development';
        confidenceColor = 'warning';
      } else {
        confidenceIcon = 'fa-heart';
        confidenceText = 'Preventive counseling recommended to maintain relationship health';
        confidenceColor = 'success';
      }
      
      const summary = `
        <div class="row mb-4">
          <div class="col-md-6 mb-3">
            <div class="card border-${riskColor} shadow-sm">
              <div class="card-body text-center">
                <h5 class="text-${riskColor} mb-3">
                  <i class="fas ${riskIcon} mr-2"></i>Risk Level
                </h5>
                <div class="mb-3">
                  <span class="badge badge-${riskColor} badge-lg" style="font-size: 1.5rem; padding: 0.75rem 1.5rem;">${riskLevel} Risk</span>
                </div>
                <p class="text-muted mb-0" style="font-size: 0.9rem;">
                  ${riskText}
                </p>
                <small class="text-muted d-block mt-2" style="font-size: 0.75rem;">
                  <i class="fas fa-brain mr-1"></i>Hybrid Assessment: Responses + Demographics + ML Patterns
                </small>
              </div>
            </div>
          </div>
          <div class="col-md-6 mb-3">
            <div class="card border-info shadow-sm">
              <div class="card-body">
                <h6 class="text-info mb-3">
                  <i class="fas fa-user-md mr-2"></i>Counseling Recommendation
                </h6>
                <div class="mb-2">
                  <span class="badge badge-${confidenceColor} badge-lg mr-2">${confidencePercentage}% Priority</span>
                </div>
                <p class="text-muted mb-0" style="font-size: 0.9rem;">
                  ${confidenceText}
                </p>
                <small class="text-muted d-block mt-2">
                  <i class="far fa-clock mr-1"></i>${new Date(analysis.generated_at || analysis.updated_at).toLocaleDateString()}
                </small>
              </div>
            </div>
          </div>
        </div>
      `;
      
      aiSummary.innerHTML = summary;
      summaryWrap.style.display = '';

      // Display ML recommendations
      cardsWrap.innerHTML = '';
      
      // Priority Categories Card - Simplified
      if (focusCategories.length > 0) {
        const categoriesCard = document.createElement('div');
        categoriesCard.className = 'mb-4';
        
        const getScoreInfo = (score) => {
          if (score > 0.6) return { color: 'danger', text: 'High Priority', barClass: 'bg-danger' };
          if (score > 0.3) return { color: 'warning', text: 'Moderate Priority', barClass: 'bg-warning' };
          return { color: 'success', text: 'Low Priority', barClass: 'bg-success' };
        };
        
        categoriesCard.innerHTML = `
          <div class="card shadow-sm">
            <div class="card-header bg-light">
              <h6 class="mb-0"><i class="fas fa-bullseye mr-2"></i>Priority Categories for Counseling</h6>
            </div>
            <div class="card-body">
              <div class="row">
                ${focusCategories.map((cat) => {
                  const scoreInfo = getScoreInfo(cat.score);
                  const scorePercentage = (cat.score * 100).toFixed(0);
                  
                  return `
                  <div class="col-md-6 mb-3">
                    <div class="card border-${scoreInfo.color}">
                      <div class="card-body">
                        <h6 class="mb-2">${cat.name}</h6>
                        <div class="mb-2">
                          <span class="badge badge-${scoreInfo.color}">${scoreInfo.text}</span>
                          <span class="badge badge-light ml-2">${scorePercentage}%</span>
                        </div>
                        <div class="progress" style="height: 20px;">
                          <div class="progress-bar ${scoreInfo.barClass}" 
                               style="width: ${scorePercentage}%">
                            ${scorePercentage}%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                `;
                }).join('')}
              </div>
            </div>
          </div>
        `;
        cardsWrap.appendChild(categoriesCard);
      }
      
      // Recommendations Card - Simplified
      const mainCard = document.createElement('div');
      mainCard.className = 'mb-4';
      
      const finalRecommendations = getPersonalizedRecommendations(mlRecommendations, focusCategories);
      
      mainCard.innerHTML = `
        <div class="card shadow-sm">
          <div class="card-header bg-light">
            <h6 class="mb-0"><i class="fas fa-clipboard-list mr-2"></i>Assessment & Recommendations</h6>
          </div>
          <div class="card-body">
            <ul class="list-unstyled mb-0">
              ${finalRecommendations.map(rec => `<li class="mb-2"><i class="fas fa-check text-primary mr-2"></i>${rec}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
      cardsWrap.appendChild(mainCard);
      
    } catch (e) {
      result.textContent = 'Error: ' + e.message;
    }
  }

  function formatAIRecommendations(topics) {
    // Format the AI-generated recommendations
    if (typeof topics === 'string') {
      // If it's a raw AI response, try to format it
      const lines = topics.split('\n').filter(line => line.trim());
      let formatted = '<ul class="list-unstyled">';
      
      lines.forEach(line => {
        if (line.trim() && !line.includes('Generate') && !line.includes('Couple profile')) {
          formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${line.trim()}</li>`;
        }
      });
      
      formatted += '</ul>';
      return formatted;
    } else if (Array.isArray(topics)) {
      // If it's an array of topics
      let formatted = '<ul class="list-unstyled">';
      topics.forEach(topic => {
        formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${topic}</li>`;
      });
      formatted += '</ul>';
      return formatted;
    } else if (topics && topics.ai_generated) {
      // Handle new AI-generated format
      const lines = topics.ai_generated.split('\n').filter(line => line.trim());
      let formatted = '<div class="ai-generated-content">';
      formatted += '<div class="alert alert-info mb-3"><i class="fas fa-robot mr-2"></i>Generated by TinyLlama AI</div>';
      formatted += '<ul class="list-unstyled">';
      
      lines.forEach(line => {
        if (line.trim() && !line.includes('Generate') && !line.includes('Couple profile')) {
          formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${line.trim()}</li>`;
        }
      });
      
      formatted += '</ul></div>';
      return formatted;
    } else if (topics && topics.topics) {
      // Handle template-based format
      let formatted = '<ul class="list-unstyled">';
      topics.topics.forEach(topic => {
        formatted += `<li class="mb-2"><i class="fas fa-arrow-right text-primary mr-2"></i>${topic}</li>`;
      });
      formatted += '</ul>';
      if (topics.focus_areas) {
        formatted += `<div class="mt-3"><small class="text-muted"><i class="fas fa-info-circle mr-1"></i>${topics.focus_areas}</small></div>`;
      }
      return formatted;
    } else {
      return '<p class="text-muted">AI recommendations are being processed...</p>';
    }
  }

  const params = new URLSearchParams(window.location.search);
  const accessIdParam = params.get('access_id');
  if (accessIdParam) renderDetail(accessIdParam);
})();
</script>

<style>
/* Simple, clean design */
.card {
  border: 1px solid #dee2e6;
  border-radius: 0.5rem;
}

.card.shadow-sm {
  box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
}

.badge-lg {
  font-size: 1.2rem;
  padding: 0.6rem 1rem;
  font-weight: 600;
}

.progress {
  border-radius: 0.25rem;
}

.progress-bar {
  font-weight: 600;
  font-size: 0.85rem;
}

.card-header.bg-light {
  background-color: #f8f9fa !important;
  border-bottom: 1px solid #dee2e6;
}

.content-wrapper {
  min-height: calc(100vh - 200px);
}

/* Responsive */
@media (max-width: 768px) {
  .badge-lg {
    font-size: 1rem;
    padding: 0.5rem 0.75rem;
  }
}
</style>
</body>
</html>
