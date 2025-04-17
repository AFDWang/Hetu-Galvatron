import os
import torch
import pytest
from collections import OrderedDict
from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_bert_mlm

@pytest.mark.model
def test_convert_checkpoints_bert_mlm(checkpoint_dir):
    # Use the checkpoint_dir fixture from conftest.py
    input_checkpoint = checkpoint_dir["baseline"]
    output_dir = checkpoint_dir["converted"]
    
    # Create mock BERT checkpoint
    model_state = OrderedDict([
        # Embedding layer parameters
        ('bert.embeddings.word_embeddings.weight', torch.randn(30522, 768)),
        ('bert.embeddings.position_embeddings.weight', torch.randn(512, 768)),
        ('bert.embeddings.token_type_embeddings.weight', torch.randn(2, 768)),
        ('bert.embeddings.LayerNorm.weight', torch.randn(768)),
        ('bert.embeddings.LayerNorm.bias', torch.randn(768)),
        
        # Layer 0 transformer parameters
        ('bert.encoder.layer.0.attention.self.query.weight', torch.randn(768, 768)),
        ('bert.encoder.layer.0.attention.self.query.bias', torch.randn(768)),
        ('bert.encoder.layer.0.attention.self.key.weight', torch.randn(768, 768)),
        ('bert.encoder.layer.0.attention.self.key.bias', torch.randn(768)),
        ('bert.encoder.layer.0.attention.self.value.weight', torch.randn(768, 768)),
        ('bert.encoder.layer.0.attention.self.value.bias', torch.randn(768)),
        ('bert.encoder.layer.0.attention.output.dense.weight', torch.randn(768, 768)),
        ('bert.encoder.layer.0.attention.output.dense.bias', torch.randn(768)),
        ('bert.encoder.layer.0.attention.output.LayerNorm.weight', torch.randn(768)),
        ('bert.encoder.layer.0.attention.output.LayerNorm.bias', torch.randn(768)),
        ('bert.encoder.layer.0.intermediate.dense.weight', torch.randn(3072, 768)),
        ('bert.encoder.layer.0.intermediate.dense.bias', torch.randn(3072)),
        ('bert.encoder.layer.0.output.dense.weight', torch.randn(768, 3072)),
        ('bert.encoder.layer.0.output.dense.bias', torch.randn(768)),
        ('bert.encoder.layer.0.output.LayerNorm.weight', torch.randn(768)),
        ('bert.encoder.layer.0.output.LayerNorm.bias', torch.randn(768)),
        
        # Pooler layer parameters
        ('bert.pooler.dense.weight', torch.randn(768, 768)),
        ('bert.pooler.dense.bias', torch.randn(768)),
        
        # MLM prediction head
        ('cls.predictions.transform.dense.weight', torch.randn(768, 768)),
        ('cls.predictions.transform.dense.bias', torch.randn(768)),
        ('cls.predictions.transform.LayerNorm.weight', torch.randn(768)),
        ('cls.predictions.transform.LayerNorm.bias', torch.randn(768)),
        ('cls.predictions.decoder.weight', torch.randn(30522, 768)),
        ('cls.predictions.bias', torch.randn(30522)),
    ])
    
    # Save mock checkpoint to input directory
    checkpoint_path = os.path.join(input_checkpoint, 'bert_model.bin')
    torch.save(model_state, checkpoint_path)
    
    # Call the function to test
    convert_checkpoints_bert_mlm(input_checkpoint, output_dir)
    
    # Verify the output directory is created correctly
    assert os.path.exists(output_dir)
    
    # Verify the per-layer files are generated correctly
    expected_files = [
        'bert_embeddings.pt',
        'bert_encoder_layer_0.pt',
        'bert_pooler.pt',
        'cls_predictions.pt'
    ]
    
    for filename in expected_files:
        file_path = os.path.join(output_dir, filename)
        assert os.path.exists(file_path), f"File {filename} was not created"
        
        # Load and verify the contents of each file
        params = torch.load(file_path)
        
        if filename == 'bert_embeddings.pt':
            # Verify embedding layer parameters
            assert 'word_embeddings.weight' in params
            assert 'position_embeddings.weight' in params
            assert 'token_type_embeddings.weight' in params
            assert 'LayerNorm.weight' in params
            assert 'LayerNorm.bias' in params
            
        elif filename == 'bert_encoder_layer_0.pt':
            # Verify transformer layer parameters
            assert 'attention.self.query.weight' in params
            assert 'attention.self.key.weight' in params
            assert 'attention.self.value.weight' in params
            assert 'attention.output.dense.weight' in params
            assert 'intermediate.dense.weight' in params
            assert 'output.dense.weight' in params
            
        elif filename == 'bert_pooler.pt':
            # Verify pooler layer parameters
            assert 'dense.weight' in params
            assert 'dense.bias' in params
            
        elif filename == 'cls_predictions.pt':
            # Verify prediction head parameters
            assert 'transform.dense.weight' in params
            assert 'decoder.weight' in params
            assert 'bias' in params